import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions.dirichlet import Dirichlet

from algorithms.base_model import BaseModel
from models.simple import MultiWayLinear
from models.iaf import IAF
from priors import sample_categorical, get_prior, GaussPrior
from ..utils import get_kld_weights, get_q_and_p, cs_entropy, get_q2
from .utils import sample, sample_from_dirichlet, sample_k


log = logging.getLogger(__name__)


def gauss_log_prob(rep, mu, log_var):
    return torch.mean(-torch.log(torch.tensor(2 * 3.14)) - (torch.sum(log_var) / 2) - (torch.sum(((rep - mu) ** 2) / log_var.exp(), -1) / 2))


def adjusted_mse_loss(reduction='mean'):
    mse = torch.nn.MSELoss(reduction=reduction)
    return lambda x, y: mse(x, y)


class VAE(BaseModel):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset.name
        self.latent_size = config.model.latent_size
        self.num_epochs = config.model.num_epochs
        self.hidden_size = config.model.hidden_size
        self.lr = config.model.lr
        self.batch_size = config.model.batch_size
        self.prior = get_prior(config)
        self.p_weight = config.model.p_weight

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Set loss
        if config.dataset.name.startswith('mnist'):
            self.criterion = torch.nn.BCELoss()
            self.criterion_nored = torch.nn.BCELoss(reduction='none')
        else:
            self.criterion = adjusted_mse_loss()
            self.criterion_nored = adjusted_mse_loss(reduction='none')

        self.kld_weights = get_kld_weights(config.model.kld_weight, 
                                           self.num_epochs, 
                                           config.model.kld_schedule, 
                                           config.model.kld_warmup)
        
        self.save_folder = config.save_folder

        self.init_models_and_optims()

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)
        
        self.Q.to(self.device)
        self.P.to(self.device)

        # Set optimizators
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), 
                                lr=self.lr)
        
    def _call_schedulers(self):
        pass

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            
            self.P.train()
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                inputs_reconstructed = self.P(rep)

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                reg_loss2 = self.prior.evaluate(rep)
                reg_loss = reg_loss1 - self.p_weight * reg_loss2
                
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss1.detach().item(), reg_loss2.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        reps = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                mu, var_s = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
            reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def get_elbo(self, dataloader):
        self.Q.eval()
        self.P.eval()
        reps = []
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]
            with torch.set_grad_enabled(False):
                mu, var_s = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                reg_loss2 = self.prior.evaluate(rep)

                inputs_reconstructed = self.P(rep)

                rec = -self.criterion(inputs_reconstructed, inputs)

            elbos.append(rec - (reg_loss1 - reg_loss2))
        # log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])


class IWAE(VAE):
    def __init__(self, config):
        self.k = config.model.k
        super().__init__(config)

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            
            self.P.train()
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                losses = []
                for _ in range(self.k):
                    rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                    reg_loss1 = log_prob

                    inputs_reconstructed = self.P(rep)

                    rec_loss = self.criterion_nored(inputs_reconstructed, inputs)
                    if self.config.dataset.name.startswith('cifar10'):
                        rec_loss = torch.mean(rec_loss, (1, 2, 3))
                    else:
                        rec_loss = torch.mean(rec_loss, 1)

                    reg_loss2 = self.prior.evaluate(rep, nored=True)
                    reg_loss = reg_loss1 - self.p_weight * reg_loss2
                    log.info(rec_loss.size(), reg_loss.size())
                    losses.append(rec_loss + self.kld_weights[self.ep] * reg_loss)

                log_w = torch.stack(losses, 1)
                log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
                # compute normalized importance weights (no gradient)
                w = log_w_minus_max.exp()
                w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
                # compute loss
                loss = (w_tilde * log_w).sum(1).mean()
                
                # loss = rec_loss + self.kld_weights[self.ep] * reg_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        reps = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                mu, var_s = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
            reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def get_elbo(self, dataloader):
        self.Q.eval()
        self.P.eval()
        reps = []
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]
            with torch.set_grad_enabled(False):
                mu, var_s = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                reg_loss2 = self.prior.evaluate(rep)

                inputs_reconstructed = self.P(rep)

                rec = -self.criterion(inputs_reconstructed, inputs)

            elbos.append(rec - (reg_loss1 - reg_loss2))
        # log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])


class ClusteringVAE(VAE):
    def __init__(self, config):
        self.n_clusters = config.model.n_clusters
        self.batch_size = config.model.batch_size
        self.eta = config.model.eta
        self.std = config.prior.gauss_std
        super().__init__(config)
        
    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.cluster_layer = nn.Linear(self.n_clusters, self.latent_size, bias=False)
        with torch.no_grad():
            weights = torch.rand(self.n_clusters, self.latent_size) * 5 - 2.5
            self.cluster_layer.weight.copy_(weights.T)
    
        self.Q.to(self.device)
        self.P.to(self.device)
        self.cluster_layer.to(self.device)

        # Set optimizators
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.cluster_layer.parameters()), 
                                lr=self.lr)
        self.optim_scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                             milestones=[100], 
                                                             gamma=0.1)

    def _call_schedulers(self):
        self.optim_scheduler.step()
    
    def _calc_cluster_loss(self, a_heads, b_heads):
        EPS = 1e-6
        a_vectors = self.cluster_layer(a_heads)
        b_vectors = self.cluster_layer(b_heads)
        # cluster_loss = torch.clip(self.eta - torch.sqrt(torch.sum((a_vectors - b_vectors) ** 2, dim=-1) + EPS), 0)
        # cluster_loss = -torch.log(torch.clip(torch.sum((a_vectors - b_vectors) ** 2, dim=-1), 0, self.eta) + EPS)
        # cluster_loss[torch.all(a_heads == b_heads, dim=1)] = 0
        # cluster_loss = torch.mean(cluster_loss)
        cluster_loss = torch.mean(torch.clip(self.eta - torch.sqrt(torch.sum((a_vectors - b_vectors) ** 2, dim=-1) + EPS), 0))
        add_loss = 100 * torch.mean(torch.clip(a_vectors - 2.5, 0)) + 100 * torch.mean(torch.clip(-a_vectors-2.5, 0)) # Make sure stays within 2.5
        return cluster_loss + add_loss

    def _get_alphas(self, h, c):
        h = F.tanh(h)
        c = F.softplus(c)
        alphas = torch.exp(h * 5.15 - (6.9 - 5.15))
        return alphas

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        EPS = 1e-6

        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            a_heads = sample_categorical(self.n_clusters, self.n_clusters).to(self.device)
            b_heads = sample_categorical(self.n_clusters, self.n_clusters).to(self.device)
            
            self.P.train()
            self.Q.train()
            self.cluster_layer.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var, h, c = self.Q(inputs)
                alphas = self._get_alphas(h, c)

                ys = sample_from_dirichlet(alphas, n_clusters=self.n_clusters) 
                ys = (ys + EPS) / torch.sum(ys + EPS, -1).view(-1, 1)
                inputs_reconstructed = self.P(sample(mu, log_var) + self.cluster_layer(ys))

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - 2 * np.log(self.std) - (mu ** 2 + log_var.exp()) / (self.std * self.std * 2), dim = 1), dim = 0)

                prior_alphas = 1e-3 * torch.ones_like(alphas)
                prior_dist = Dirichlet(prior_alphas) 
                q_dist = Dirichlet(alphas)
                dir_reg_loss = torch.mean(-torch.clamp(prior_dist.log_prob(ys), min=-100, max=170) + torch.clamp(q_dist.log_prob(ys), min=-100, max=170))
                if torch.min(q_dist.log_prob(ys)) < -100 or torch.max(q_dist.log_prob(ys)) > 170 or torch.min(prior_dist.log_prob(ys)) < -100 or torch.max(prior_dist.log_prob(ys)) > 170:
                    log.info(f"Q Clamped {torch.min(q_dist.log_prob(ys))} {torch.max(q_dist.log_prob(ys))}")
                    log.info(f"prior Clamped {torch.min(prior_dist.log_prob(ys))} {torch.max(prior_dist.log_prob(ys))}")

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + 0.5 * self.kld_weights[self.ep] * dir_reg_loss
                loss.backward()

                self.optim.step()

            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                cluster_loss = self._calc_cluster_loss(a_heads, b_heads)
                cluster_loss.backward()
                self.optim.step()
            
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), cluster_loss.detach().item(), dir_reg_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l

    def eval(self, dataloader):
        self.Q.eval()
        self.cluster_layer.eval()
        reps = []
        all_ys = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                mu, log_var, h, c = self.Q(inputs)
                alphas = self._get_alphas(h, c)
                ys = sample_from_dirichlet(alphas, n_clusters=self.n_clusters) 
                ys = ys / torch.sum(ys, -1).view(-1, 1)
                rep = sample(mu, log_var) + self.cluster_layer(ys)
            reps.append(rep.cpu().numpy())
            all_ys.append(ys.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(all_ys, axis=0)
    
    def eval_label(self, dataloader):
        _, ys = self.eval(dataloader)
        return torch.argmax(torch.tensor(ys), -1).cpu().numpy()
    
    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            'cluster_layer': self.cluster_layer.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])
        self.cluster_layer.load_state_dict(checkpoint['cluster_layer'])


class IAF_VAE(VAE):
    def __init__(self, config):
        self.context_size = config.model.context_size
        self.prior = get_prior(config)
        self.p_weight = config.model.p_weight
        super().__init__(config)
        
    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.iaf_layers = torch.nn.ModuleList([IAF(self.latent_size, self.context_size, parity=i % 2) for i in range(4)])
        
        self.Q.to(self.device)
        self.P.to(self.device)
        self.iaf_layers.to(self.device)

        # Set optimizators
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.iaf_layers.parameters()), 
                                lr=self.lr)
        
    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]
            
            self.P.train()
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld
                inputs_reconstructed = self.P(rep)

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = self.prior.evaluate(rep)

                reg_loss = reg_loss1 - reg_loss2 - self.p_weight * reg_loss3

                # log.info(reg_loss1)
                # log.info(reg_loss2)
                # log.info(reg_loss3)
                
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss1.detach().item(), reg_loss2.detach().item(), reg_loss3.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        reps = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
                for layer in self.iaf_layers: 
                    rep, _ = layer(rep, context)
            reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)
    
    def get_elbo(self, dataloader):
        self.Q.eval()
        self.P.eval()
        reps = []
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]
            with torch.set_grad_enabled(False):
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = self.prior.evaluate(rep)


                inputs_reconstructed = self.P(rep)

                rec = -self.criterion(inputs_reconstructed, inputs)

            elbos.append(rec + reg_loss3 - (reg_loss1 - reg_loss2))
        # log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            'iaf_layers': self.iaf_layers.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])
        self.iaf_layers.load_state_dict(checkpoint['iaf_layers'])


class VAE_semi_simple(VAE):
    def __init__(self, config):
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_classes = config.dataset.n_classes
        self.cls_weight = config.model.cls_weight
        super().__init__(config)

    def init_models_and_optims(self):
        super().init_models_and_optims()

        self.classifier = MultiWayLinear(2, [128, 128, 128], 10)
        self.classifier.to(self.device)

        self.classifier_optim = optim.Adam(self.classifier.parameters(), lr=self.lr)

        self.classifier_train()

    def classifier_train(self):
        self.classifier.train()
        log.info('Training classifier')
        for i in range(50000):
            self.classifier_optim.zero_grad()
            prior_samples, classes = self.prior.sample(self.batch_size, with_classes=True)
            with torch.set_grad_enabled(True):
                class_logits = self.classifier(prior_samples)
                class_loss = self.ce(class_logits, classes)
                class_loss.backward()
                self.classifier_optim.step()
            if i % 5000 == 0:
                log.info(f'Step {i} Loss {class_loss}')

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.P.train()
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                inputs_reconstructed = self.P(rep)

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                reg_loss2 = self.prior.evaluate(rep)
                reg_loss = reg_loss1 - self.p_weight * reg_loss2

                class_logits = self.classifier(rep)
                cls_loss = self.ce(class_logits[targets_cpu != self.n_classes], targets[targets_cpu != self.n_classes])
                
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.cls_weight * self.kld_weights[self.ep] * cls_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss1.detach().item(), reg_loss2.detach().item(), cls_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l


class IAF_VAE_semi_simple(IAF_VAE):
    def __init__(self, config):
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_classes = config.dataset.n_classes
        self.cls_weight = config.model.cls_weight
        super().__init__(config)

    def init_models_and_optims(self):
        super().init_models_and_optims()

        self.classifier = MultiWayLinear(2, [128, 128, 128], 10)
        self.classifier.to(self.device)

        self.classifier_optim = optim.Adam(self.classifier.parameters(), lr=self.lr)

        self.classifier_train()

    def classifier_train(self):
        self.classifier.train()
        log.info('Training classifier')
        for i in range(50000):
            self.classifier_optim.zero_grad()
            prior_samples, classes = self.prior.sample(self.batch_size, with_classes=True)
            with torch.set_grad_enabled(True):
                class_logits = self.classifier(prior_samples)
                class_loss = self.ce(class_logits, classes)
                class_loss.backward()
                self.classifier_optim.step()
            if i % 5000 == 0:
                log.info(f'Step {i} Loss {class_loss}')
        
    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            batch_size = inputs.size()[0]
            
            self.P.train()
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld
                inputs_reconstructed = self.P(rep)

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = self.prior.evaluate(rep)

                reg_loss = reg_loss1 - reg_loss2 - self.p_weight * reg_loss3

                # log.info(reg_loss1)
                # log.info(reg_loss2)
                # log.info(reg_loss3)
                class_logits = self.classifier(rep)
                cls_loss = self.ce(class_logits[targets_cpu != self.n_classes], targets[targets_cpu != self.n_classes])
                
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.cls_weight * self.kld_weights[self.ep] * cls_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss1.detach().item(), reg_loss2.detach().item(), reg_loss3.detach().item(), cls_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    

class VAE_semi(VAE):
    def __init__(self, config):
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_classes = config.dataset.n_classes
        self.cls_weight = config.model.cls_weight
        self.q_norm = config.model.q_norm
        super().__init__(config)

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets_cpu = targets
            targets = targets.to(self.device)
            
            self.P.train()
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s, cs_logits = self.Q(inputs)
                cs = F.softmax(cs_logits, 1)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                inputs_reconstructed = self.P(rep)

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                unlabeled_indices = torch.arange(inputs.size(0))[targets_cpu == self.n_classes]
                unlabeled_loss = torch.zeros(1).to(self.device)
                labeled_loss = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    unlabeled_loss = unlabeled_loss + torch.sum(cs[unlabeled_indices, i] * self.prior.evaluate_partition(rep[unlabeled_indices], i))
                    labeled_loss = labeled_loss + torch.sum(self.prior.evaluate_partition(rep[targets_cpu == i], i))
                reg_loss2 = unlabeled_loss / len(unlabeled_indices) + labeled_loss / (max(1, inputs.size(0) - len(unlabeled_indices))) + 0.001 * cs_entropy(cs[unlabeled_indices])
                
                reg_loss = reg_loss1 - self.p_weight * reg_loss2

                if len(targets[targets_cpu != self.n_classes]) == 0:
                    cls_loss = torch.zeros(1).to(self.device)
                else:
                    cls_loss = self.ce(cs_logits[targets_cpu != self.n_classes], targets[targets_cpu != self.n_classes])
                
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.cls_weight * self.kld_weights[self.ep] * cls_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss1.detach().item(), reg_loss2.detach().item(), cls_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        reps = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                mu, var_s, _ = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
            reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def get_elbo(self, dataloader):
        self.Q.eval()
        self.P.eval()
        reps = []
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]
            with torch.set_grad_enabled(False):
                mu, var_s, cs_logits = self.Q(inputs)
                cs = F.softmax(cs_logits, 1)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                reg_loss2 = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    reg_loss2 = reg_loss2 + torch.sum(cs[:,i] * self.prior.evaluate_partition(rep, i))
                reg_loss2 = reg_loss2 / batch_size

                inputs_reconstructed = self.P(rep)

                rec = -self.criterion(inputs_reconstructed, inputs)

            elbos.append(rec - (reg_loss1 - reg_loss2))
        # log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()
    

class IAF_VAE_semi(IAF_VAE):
    def __init__(self, config):
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_classes = config.dataset.n_classes
        self.cls_weight = config.model.cls_weight
        self.q_norm = config.model.q_norm
        super().__init__(config)

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets_cpu = targets
            targets = targets.to(self.device)

            batch_size = inputs.size()[0]
            
            self.P.train()
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s, context, cs_logits = self.Q(inputs)
                cs = F.softmax(cs_logits, 1)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld
                inputs_reconstructed = self.P(rep)

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                reg_loss2 = torch.mean(log_det)

                unlabeled_indices = torch.arange(inputs.size(0))[targets_cpu == self.n_classes]
                unlabeled_loss = torch.zeros(1).to(self.device)
                labeled_loss = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    unlabeled_loss = unlabeled_loss + torch.sum(cs[unlabeled_indices, i] * self.prior.evaluate_partition(rep[unlabeled_indices], i))
                    labeled_loss = labeled_loss + torch.sum(self.prior.evaluate_partition(rep[targets_cpu == i], i))
                reg_loss3 = unlabeled_loss / len(unlabeled_indices) + labeled_loss / (max(1, inputs.size(0) - len(unlabeled_indices))) + 0.001 * cs_entropy(cs[unlabeled_indices])

                reg_loss = reg_loss1 - reg_loss2 - self.p_weight * reg_loss3

                if len(targets[targets_cpu != self.n_classes]) == 0:
                    cls_loss = torch.zeros(1).to(self.device)
                else:
                    cls_loss = self.ce(cs_logits[targets_cpu != self.n_classes], targets[targets_cpu != self.n_classes])
                
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.cls_weight * self.kld_weights[self.ep] * cls_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss1.detach().item(), reg_loss2.detach().item(), reg_loss3.detach().item(), cls_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        reps = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                mu, var_s, context, _ = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
                for layer in self.iaf_layers: 
                    rep, _ = layer(rep, context)
            reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)
    
    def get_elbo(self, dataloader):
        self.Q.eval()
        self.P.eval()
        reps = []
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]
            with torch.set_grad_enabled(False):
                mu, var_s, context, cs_logits = self.Q(inputs)
                cs = F.softmax(cs_logits, 1)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    reg_loss3 = reg_loss3 + torch.sum(cs[:,i] * self.prior.evaluate_partition(rep, i))
                reg_loss3 = reg_loss3 / batch_size


                inputs_reconstructed = self.P(rep)

                rec = -self.criterion(inputs_reconstructed, inputs)

            elbos.append(rec + reg_loss3 - (reg_loss1 - reg_loss2))
        # log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()

class ClusteringIAFVAE(ClusteringVAE):
    def __init__(self, config):
        self.context_size = config.model.context_size
        self.prior = get_prior(config)
        self.p_weight = config.model.p_weight
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.cluster_layer = nn.Linear(self.n_clusters, self.latent_size, bias=False)
        with torch.no_grad():
            weights = torch.rand(self.n_clusters, self.latent_size) * 5 - 2.5
            self.cluster_layer.weight.copy_(weights.T)

        self.iaf_layers = torch.nn.ModuleList([IAF(self.latent_size, self.context_size, parity=i % 2) for i in range(4)])
    
        self.Q.to(self.device)
        self.P.to(self.device)
        self.cluster_layer.to(self.device)
        self.iaf_layers.to(self.device)

        # Set optimizators
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.cluster_layer.parameters()) + list(self.iaf_layers.parameters()), 
                                lr=self.lr)
        self.optim_scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                             milestones=[100], 
                                                             gamma=0.1)

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        EPS = 1e-6

        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            a_heads = sample_categorical(self.n_clusters, self.n_clusters).to(self.device)
            b_heads = sample_categorical(self.n_clusters, self.n_clusters).to(self.device)
            
            self.P.train()
            self.Q.train()
            self.cluster_layer.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, var_s, h, c, context = self.Q(inputs)
                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = self.prior.evaluate(rep)

                reg_loss = reg_loss1 - reg_loss2 - self.p_weight * reg_loss3

                alphas = self._get_alphas(h, c)
                ys = sample_from_dirichlet(alphas, n_clusters=self.n_clusters) 
                ys = (ys + EPS) / torch.sum(ys + EPS, -1).view(-1, 1)
                inputs_reconstructed = self.P(rep + self.cluster_layer(ys))

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                prior_alphas = 1e-3 * torch.ones_like(alphas)
                prior_dist = Dirichlet(prior_alphas) 
                q_dist = Dirichlet(alphas)
                dir_reg_loss = torch.mean(-torch.clamp(prior_dist.log_prob(ys), min=-100, max=170) + torch.clamp(q_dist.log_prob(ys), min=-100, max=170))
                if torch.min(q_dist.log_prob(ys)) < -100 or torch.max(q_dist.log_prob(ys)) > 170 or torch.min(prior_dist.log_prob(ys)) < -100 or torch.max(prior_dist.log_prob(ys)) > 170:
                    log.info(f"Q Clamped {torch.min(q_dist.log_prob(ys))} {torch.max(q_dist.log_prob(ys))}")
                    log.info(f"prior Clamped {torch.min(prior_dist.log_prob(ys))} {torch.max(prior_dist.log_prob(ys))}")

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + 0.5 * self.kld_weights[self.ep] * dir_reg_loss
                loss.backward()

                self.optim.step()

            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                cluster_loss = self._calc_cluster_loss(a_heads, b_heads)
                cluster_loss.backward()
                self.optim.step()
            
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), cluster_loss.detach().item(), dir_reg_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l

    def eval(self, dataloader):
        self.Q.eval()
        self.cluster_layer.eval()
        reps = []
        all_ys = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                mu, var_s, h, c, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
                for layer in self.iaf_layers: 
                    rep, _ = layer(rep, context)
                
                alphas = self._get_alphas(h, c)
                ys = sample_from_dirichlet(alphas, n_clusters=self.n_clusters) 
                ys = ys / torch.sum(ys, -1).view(-1, 1)
                rep = rep + self.cluster_layer(ys)
            reps.append(rep.cpu().numpy())
            all_ys.append(ys.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(all_ys, axis=0)
    
    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            'cluster_layer': self.cluster_layer.state_dict(),
            'iaf_layers': self.iaf_layers.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])
        self.cluster_layer.load_state_dict(checkpoint['cluster_layer'])
        self.iaf_layers.load_state_dict(checkpoint['iaf_layers'])


class H_VAE(VAE):
    def __init__(self, config):
       raise NotImplementedError


class H_IAF_VAE(VAE):
    def __init__(self, config):
        self.context_size = config.model.context_size
        self.prior = get_prior(config)
        self.gauss_prior = GaussPrior(config.model.latent_size)
        self.p_weight = config.model.p_weight
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.Q2 = get_q2(self.config)

        self.iaf_layers = torch.nn.ModuleList([IAF(self.latent_size, self.context_size, parity=i % 2, nh=64) for i in range(4)])

        self.iaf_layers_2 = torch.nn.ModuleList([IAF(self.latent_size, self.context_size, parity=i % 2, nh=64) for i in range(4)])
        
        self.Q.to(self.device)
        self.P.to(self.device)
        self.iaf_layers.to(self.device)
        self.Q2.to(self.device)
        self.iaf_layers_2.to(self.device)

        # Set optimizators
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.iaf_layers.parameters()) + list(self.Q2.parameters()) + list(self.iaf_layers_2.parameters()), 
                                lr=self.lr)
        
    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]
            
            self.P.train()
            self.Q.train()
            self.Q2.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                # -------------- Q1 -------------------
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = self.gauss_prior.evaluate(rep)
                # -----------------------------------

                # -------------- Q2 -------------------
                mu, var_s, context = self.Q2(inputs, rep)

                var = torch.clip(torch.sigmoid(var_s), 1e-6, 1)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss4 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers_2: 
                    rep, ld = layer(rep, context)
                    log_det += ld

                reg_loss5 = torch.mean(log_det)
                reg_loss6 = self.prior.evaluate(rep)
                # --------------------------------------

                inputs_reconstructed = self.P(rep)

                rec_loss = self.criterion(inputs_reconstructed, inputs)

                reg_loss = (reg_loss1 - reg_loss2 - self.p_weight * reg_loss3) + (reg_loss4 - reg_loss5 - self.p_weight * reg_loss6)
                
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss
                loss.backward()
                self.optim.step()

            res = np.asarray([rec_loss.detach().item(), reg_loss1.detach().item(), reg_loss2.detach().item(), reg_loss3.detach().item(), reg_loss4.detach().item(), reg_loss5.detach().item(), reg_loss6.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.Q2.eval()
        reps = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                # --------- Q1 ------------
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
                for layer in self.iaf_layers: 
                    rep, _ = layer(rep, context)
                # -------------------------

                # --------- Q2 ------------
                mu, var_s, context = self.Q2(inputs, rep)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
                for layer in self.iaf_layers_2: 
                    rep, _ = layer(rep, context)
                # --------------------------
                
            reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def get_elbo(self, dataloader):
        self.Q.eval()
        self.Q2.eval()
        reps = []
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]
            with torch.set_grad_enabled(False):
                # -------------- Q1 -------------------
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss1 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers: 
                    rep, ld = layer(rep, context)
                    log_det += ld

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = self.gauss_prior.evaluate(rep)
                # -----------------------------------

                # -------------- Q2 -------------------
                mu, var_s, context = self.Q2(inputs, rep)

                var = torch.clip(torch.sigmoid(var_s), 1e-6, 1)
                log_var = torch.log(var)

                rep, log_prob = sample(mu, log_var, ret_log_prob=True)
                reg_loss4 = torch.mean(log_prob)

                log_det = torch.zeros(batch_size).to(self.device)
                for layer in self.iaf_layers_2: 
                    rep, ld = layer(rep, context)
                    log_det += ld

                reg_loss5 = torch.mean(log_det)
                reg_loss6 = self.prior.evaluate(rep)
                # --------------------------------------

                inputs_reconstructed = self.P(rep)

                rec = -self.criterion(inputs_reconstructed, inputs)

            elbos.append(rec + reg_loss3 - (reg_loss1 - reg_loss2) + reg_loss6 - (reg_loss4 - reg_loss5))
        # log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()
                

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            'iaf_layers': self.iaf_layers.state_dict(),
            'Q2': self.Q2.state_dict(),
            'iaf_layers_2': self.iaf_layers_2.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])
        self.iaf_layers.load_state_dict(checkpoint['iaf_layers'])
        self.Q2.load_state_dict(checkpoint['Q2'])
        self.iaf_layers_2.load_state_dict(checkpoint['iaf_layers_2'])