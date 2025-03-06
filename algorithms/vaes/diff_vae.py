import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from torch.distributions.dirichlet import Dirichlet

from models.simple import ConditionalModel, MultiWayLinear, TimeClassifier, ConditionalModelWAUX
from models.transposed_conv_net import BareConvNet
from models.iaf import IAF, ConditionalMADE, StackedMADE
from diffusion.gaussian import GaussianDiffusionNoise, TwoDiffusion
from diffusion.guidance import GuidedDiffWrapper
from utils import visualize_latent
from priors import sample_categorical, get_prior
from .vanilla_vae import VAE
from .utils import sample, DiffusionConfig, sample_from_dirichlet
from ..train_diffusion import StandardDiffusion
from ..train_classifier_guidance import ClassifierGuidance
from ..utils import AddNoise, get_q_and_p, cs_entropy, gauss_logpdf_samevalue, get_kld_weights

log = logging.getLogger(__name__)


class ClusteringPrior:
    def __init__(self, prior, cluster_layer, covariance_layer, n_clusters):
        self.prior = prior
        self.cluster_layer = cluster_layer
        self.covariance_layer = covariance_layer
        self.n_clusters = n_clusters
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def sample(self, batch_size):
        heads = sample_categorical(batch_size, self.n_clusters).to(self.device)
        # log.info(F.relu(self.covariance_layer(heads)))
        # return self.prior.sample(batch_size) * self.scale * torch.abs(self.covariance_layer(heads)) + self.cluster_layer(heads)
        return self.prior.sample(batch_size) + self.cluster_layer(heads)
    

class PartitionPrior: 
    def __init__(self, prior, partition):
        self.prior = prior
        self.partition = partition

    def sample(self, batch_size):
        return self.prior.sample(batch_size, g_classes=np.asarray([self.partition] * batch_size))

    
def get_weights(n_clusters, latent_size):
    if n_clusters == 1:
        return torch.zeros(n_clusters, latent_size)
    space = 8
    cols = int(np.sqrt(n_clusters))
    rows = -(-n_clusters // cols)
    scale = 4 / ((max(cols, rows) - 1) * space)

    means = torch.zeros(n_clusters, latent_size)
    for i in range(n_clusters):
        x = ((i // cols) * space) - ((rows - 1) * space) // 2
        y = ((i % cols) * space) - ((cols - 1) * space) // 2
        scaled_x = scale * x
        scaled_y = scale * y
        means[i] = torch.Tensor(np.concatenate(([scaled_x, scaled_y], np.zeros(latent_size - 2))))
    return means

class DiffVAE(VAE):
    def __init__(self, config):
        self.g_path = config.g_path
        self.timesteps = config.model.timesteps
        self.prior = AddNoise(get_prior(config), config.model.noise_sigma)
        self.batch_size = config.model.batch_size
        self.endtoend = config.model.endtoend
        self.diff_weight = config.model.diff_weight
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        if self.g_path is None:
            # Warm up
            self.G = self._train_diff()
        else:
            self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)
            checkpoint = torch.load(os.path.join(self.g_path, f'diff_model.pt'))
            self.G.load_state_dict(checkpoint['Q'])
        # # TODO change back
        # self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
        #                                 self.timesteps,
        #                                 self.latent_size,
        #                                 device=self.device)
        self.G.eval()

        if self.endtoend:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.G.parameters()), 
                                lr=self.lr)
        else:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), 
                                lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                        milestones=[150], 
                                                        gamma=0.1)

        self.Q.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)

    def _train_diff(self):
        diff_config = DiffusionConfig
        diff_config.num_epochs = 100
        diff_config.prior = self.prior
        diff_config.latent_size = self.latent_size
        diff_config.timesteps = self.timesteps
        diff_config.save_folder = os.path.join(self.save_folder, 'diff')
        diff = StandardDiffusion(diff_config)
        diff.train()
        diff.save(os.path.join(self.save_folder, 'diff_model.pt'))
        return diff.Q
    
    def _call_schedulers(self):
        self.scheduler.step()
        
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
            self.G.train()
            prior_samples = self.prior.sample(self.batch_size)
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var = self.Q(inputs)
                x = sample(mu, log_var)
                z = self.G.sample(batch_size, x=x)[-1]
                inputs_reconstructed = self.P(z)

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                if self.endtoend:
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    diff_loss = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")
                else:
                    diff_loss = torch.zeros(1).to(self.device)

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.diff_weight * diff_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), diff_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.G.eval()
        reps = []
        xs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var = self.Q(inputs)
                x = sample(mu, log_var)
                rep = self.G.sample(batch_size, x=x)[-1]
            reps.append(rep.cpu().numpy())
            xs.append(x.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(xs, axis=0)
    
    def get_elbo(self, dataloader):
        self.Q.eval()
        self.G.eval()
        self.P.eval()
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var = self.Q(inputs)
                y = sample(mu, log_var)
                zs = self.G.sample(batch_size, x=y, get_xt=True)
                z = zs[-1]
                inputs_reconstructed = self.P(z)

                # E_{q(y, z| x)} [log p(x|z)]
                rec = -self.criterion(inputs_reconstructed, inputs)

                # \KL (q(y_T|x) || r(y_T|y_0)) ==> \KL (q(y|x) || N(0, 1))
                prior_y = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                # \KL (q(y_0|y_1, x) || p(y_0)) where y_0 = z
                variance_t0 = self.G.posterior_variance[0]
                # prior_z = torch.mean(gauss_logpdf_samevalue(torch.sqrt(variance_t0)) - self.prior.evaluate(z))
                prior_z = torch.mean(-self.prior.evaluate(z)) # variance = 0 so pdf = 1 and logpdf = 0


                # \sum_{t=2}^T \KL (q(y_{t-1}|y_, x) || r(y_{t-1}|y_t, y_0)) where y_0 = z
                sm_kl = torch.zeros(1).to(self.device)
                for tt in range(1, self.timesteps):
                    t = torch.full((batch_size,), tt, device=self.device, dtype=torch.long)
                    sm_kl = sm_kl + self.G.kl_divergence_at_t(zs[-1], zs[-1-(tt+1)], t, zs[-1-tt])
            
            elbos.append(rec - prior_y - prior_z - sm_kl)
        log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'G': self.G.state_dict(),
            'P': self.P.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.G.load_state_dict(checkpoint['G'])
        self.P.load_state_dict(checkpoint['P'])

    def generate(self, n):
        all_outputs = []
        for i in range(0, n, self.batch_size):
            samples = self.prior.sample(min(self.batch_size, n-i))
            outputs = self.P(samples)
            all_outputs.append(outputs.cpu().detach().numpy())
        return np.concatenate(all_outputs, axis=0)
    

class DiffVAE_semi_simple(DiffVAE):
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
            self.G.train()
            prior_samples = self.prior.sample(self.batch_size)
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var = self.Q(inputs)
                x = sample(mu, log_var)
                z = self.G.sample(batch_size, x=x)[-1]
                inputs_reconstructed = self.P(z)

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                if self.endtoend:
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    diff_loss = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")
                else:
                    diff_loss = torch.zeros(1).to(self.device)

                class_logits = self.classifier(z)
                cls_loss = self.ce(class_logits[targets_cpu != self.n_classes], targets[targets_cpu != self.n_classes])

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.diff_weight * diff_loss + self.cls_weight * self.kld_weights[self.ep] * cls_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), diff_loss.detach().item(), cls_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l


class AutoClusteringDiffVAE(DiffVAE):
    def __init__(self, config):
        self.n_clusters = config.model.n_clusters
        self.batch_size = config.model.batch_size
        self.eta = config.model.eta
        # self.std = config.prior.gauss_std
        self.diff_ct = config.model.diff_ct
        self.learnable_heads = config.model.learnable_heads
        self.g_norm = config.model.g_norm
        super().__init__(config)

    def _call_schedulers(self):
        pass

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.cluster_layer = nn.Linear(self.n_clusters, self.latent_size, bias=False)
        self.cluster_layer.to(self.device)
        with torch.no_grad():
            weights = torch.rand(self.n_clusters, self.latent_size) * 0.1
            # weights = get_weights(self.n_clusters, self.latent_size)
            self.cluster_layer.weight.copy_(weights.T)

        self.covariance_layer = nn.Linear(self.n_clusters, 1, bias=False)
        self.covariance_layer.to(self.device)
        # with torch.no_grad():
        #     weights = torch.ones(self.n_clusters, self.latent_size)
        #     self.covariance_layer.weight.copy_(weights.T)

        self.prior = ClusteringPrior(self.prior, self.cluster_layer, self.covariance_layer, self.n_clusters) # IMPORTANT: override prior

        if self.g_path is None:
            self.G = self._train_diff()
        else:
            self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)
            checkpoint = torch.load(self.g_path)
            self.G.load_state_dict(checkpoint['Q'])
        
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.G.parameters()), lr=self.lr)
        self.diff_optim = optim.Adam(list(self.G.parameters()), lr=0.001)
        self.cluster_optim = optim.Adam(list(self.cluster_layer.parameters()) + list(self.covariance_layer.parameters()), lr=0.01) # TODO: What learning rate is best?
        # self.cluster_optim = optim.Adam(list(self.covariance_layer.parameters()), lr=0.1)
        
        self.Q.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)

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
    
    def _update_clusters(self, z, inputs): # TODO: optimize this (might not need separate pass)
        z_in = z.detach().requires_grad_(True)
        with torch.set_grad_enabled(True):
            inputs_reconstructed = self.P(z_in)
            rec_loss = self.criterion(inputs_reconstructed, inputs)
        grads = torch.autograd.grad(rec_loss, z_in)[0]

        heads = self.cluster_layer.weight.detach().T
        mse = torch.sum((heads.view(1, -1, self.latent_size) - z.view(-1, 1, self.latent_size)) ** 2, -1)
        dist, membership = torch.min(mse, -1)
        # membership[dist > 2 * self.std * 2 * self.std] = -1
        # log.info(f'Membership {membership}')
        for i in range(self.n_clusters):
            if len(grads[membership == i]) > 0:
                heads[i] = heads[i] + torch.sum(grads[membership == i], 0) # TODO: change weight
        
        with torch.no_grad():
            self.cluster_layer.weight.copy_(heads.T)
        
    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''
        log.info(f'Cluster layer weight {self.cluster_layer.weight}')
        log.info(f'Covarance layer weight {self.covariance_layer.weight}')
        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]
            
            self.P.train()
            self.Q.train()
            self.G.eval()
            with torch.set_grad_enabled(False):
                prior_samples = self.prior.sample(self.batch_size)
            self.optim.zero_grad()
            if self.learnable_heads:
                self.cluster_optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var = self.Q(inputs)
                x = sample(mu, log_var)
                z = self.G.sample(batch_size, x=x)[-1]
                inputs_reconstructed = self.P(z)

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.g_norm)
                self.optim.step()
                if self.learnable_heads:
                    self.cluster_optim.step()

            # self._update_clusters(z, inputs)

            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item()])
            all_res += res

        # for _ in range(self.diff_ct):
        if self.learnable_heads:
            for _ in range((self.ep + 1) * 1):
                a_heads = sample_categorical(self.batch_size, self.n_clusters).to(self.device)
                b_heads = sample_categorical(self.batch_size, self.n_clusters).to(self.device)
                self.G.train()
                self.cluster_optim.zero_grad()
                with torch.set_grad_enabled(True):
                    prior_samples = self.prior.sample(self.batch_size)
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()

                    diff_loss = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")

                    cluster_loss = self._calc_cluster_loss(a_heads, b_heads)

                    if self.ep <= 100:
                        cluster_w = 1
                    else:
                        cluster_w = ((self.ep - 100) // 10) + 1

                    loss = cluster_w * cluster_loss + diff_loss
                    # loss = cluster_loss + diff_loss
                    loss.backward()
                    self.cluster_optim.step()
        
        for _ in range((self.ep + 1) * 1):
            self.G.train()
            self.diff_optim.zero_grad()
            with torch.set_grad_enabled(True):
                prior_samples = self.prior.sample(self.batch_size)
                t = torch.randint(
                    0, self.timesteps, (self.batch_size,), device=self.device
                ).long()

                diff_loss = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")

                loss = diff_loss
                loss.backward()
                self.diff_optim.step()
        if self.ep % 5 == 0: 
            with torch.no_grad():
                visualize_latent(self.prior.sample(2500).detach().cpu().numpy(), os.path.join(self.save_folder, f'prior_latents_ep{self.ep}.png'))
                g_samples = []
                for _ in range(20):
                    g_samples.append(self.G.sample(128, x=torch.randn(128, 2).to(self.device))[-1])
                g_samples = torch.cat(g_samples, 0)
                visualize_latent(g_samples.detach().cpu().numpy(), os.path.join(self.save_folder, f'g_latents_ep{self.ep}.png'))
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.G.eval()
        reps = []
        xs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var = self.Q(inputs)
                x = sample(mu, log_var)
                rep = self.G.sample(batch_size, x=x)[-1]
            reps.append(rep.cpu().numpy())
            xs.append(x.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(xs, axis=0)

    def eval_label(self, dataloader):
        z, _ = self.eval(dataloader)
        z = torch.tensor(z, device=self.device)
        heads = self.cluster_layer.weight.detach().T
        mse = torch.sum((heads.view(1, -1, self.latent_size) - z.view(-1, 1, self.latent_size)) ** 2, -1)
        membership = torch.argmin(mse, -1)
        return membership.cpu().numpy()

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'G': self.G.state_dict(),
            'P': self.P.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.G.load_state_dict(checkpoint['G'])
        self.P.load_state_dict(checkpoint['P'])


class ClusteringDiffVAE(VAE):
    def __init__(self, config):
        self.n_clusters = config.model.n_clusters
        self.batch_size = config.model.batch_size
        self.eta = config.model.eta

        self.g_path = config.g_path
        self.timesteps = config.model.timesteps
        self.prior = AddNoise(get_prior(config), config.model.noise_sigma)
        self.endtoend = config.model.endtoend
        self.diff_weight = config.model.diff_weight
        self.discrete_decode = config.model.discrete_decode
        super().__init__(config)
        
    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.cluster_layer = nn.Linear(self.n_clusters, self.latent_size, bias=False)
        with torch.no_grad():
            weights = torch.rand(self.n_clusters, self.latent_size) * 5 - 2.5
            self.cluster_layer.weight.copy_(weights.T)

        if self.g_path is None:
            self.G = self._train_diff()
        else:
            self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)
            checkpoint = torch.load(self.g_path)
            self.G.load_state_dict(checkpoint['Q'])
    
        self.Q.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)
        self.cluster_layer.to(self.device)

        # Set optimizator
        if self.endtoend:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.cluster_layer.parameters()) + list(self.G.parameters()), 
                                lr=self.lr)
        else:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.cluster_layer.parameters()), 
                                lr=self.lr)
        self.optim_scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                             milestones=[100], 
                                                             gamma=0.1)

    def _train_diff(self):
        diff_config = DiffusionConfig
        diff_config.prior = self.prior
        diff_config.latent_size = self.latent_size
        diff_config.timesteps = self.timesteps
        diff_config.save_folder = os.path.join(self.save_folder, 'diff')
        diff = StandardDiffusion(diff_config)
        diff.train()
        diff.save(os.path.join(self.save_folder, 'diff_model.pt'))
        return diff.Q

    def _call_schedulers(self):
        self.optim_scheduler.step()

    def _calc_cluster_loss(self, a_heads, b_heads):
        EPS = 1e-6
        a_vectors = self.cluster_layer(a_heads)
        b_vectors = self.cluster_layer(b_heads)
        # cluster_loss = torch.clip(self.eta - torch.sqrt(torch.sum((a_vectors - b_vectors) ** 2, dim=-1) + EPS), 0)
        cluster_loss = -torch.log(torch.clip(torch.sum((a_vectors - b_vectors) ** 2, dim=-1), 0, self.eta) / self.eta + EPS)
        cluster_loss[torch.all(a_heads == b_heads, dim=1)] = 0
        cluster_loss = torch.mean(cluster_loss)
        return cluster_loss

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
            batch_size = inputs.size()[0]

            a_heads = sample_categorical(self.n_clusters, self.n_clusters).to(self.device)
            b_heads = sample_categorical(self.n_clusters, self.n_clusters).to(self.device)
            
            self.P.train()
            self.Q.train()
            self.G.train()
            self.cluster_layer.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var, h, c = self.Q(inputs)
                alphas = self._get_alphas(h, c)

                ys = sample_from_dirichlet(alphas, n_clusters=self.n_clusters) 
                ys = (ys + EPS) / torch.sum(ys + EPS, -1).view(-1, 1)
                inputs_reconstructed = self.P(self.G.sample(batch_size, x=sample(mu, log_var))[-1] + self.cluster_layer(ys))

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                prior_alphas = 1e-3 * torch.ones_like(alphas)
                prior_dist = Dirichlet(prior_alphas) 
                q_dist = Dirichlet(alphas)
                dir_reg_loss = torch.mean(-torch.clamp(prior_dist.log_prob(ys), min=-80, max=120) + torch.clamp(q_dist.log_prob(ys), min=-80, max=120))
                if torch.min(q_dist.log_prob(ys)) < -100 or torch.max(q_dist.log_prob(ys)) > 170 or torch.min(prior_dist.log_prob(ys)) < -100 or torch.max(prior_dist.log_prob(ys)) > 170:
                    log.info(f"Q Clamped {torch.min(q_dist.log_prob(ys))} {torch.max(q_dist.log_prob(ys))}")
                    log.info(f"prior Clamped {torch.min(prior_dist.log_prob(ys))} {torch.max(prior_dist.log_prob(ys))}")

                if self.endtoend:
                    prior_samples = self.prior.sample(self.batch_size)
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    diff_loss = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")
                else:
                    diff_loss = torch.zeros(1).to(self.device)

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + 0.5 * self.kld_weights[self.ep] * dir_reg_loss + self.diff_weight * diff_loss
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
        self.G.eval()
        reps = []
        all_ys = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]
            with torch.set_grad_enabled(False):
                mu, log_var, h, c = self.Q(inputs)
                alphas = self._get_alphas(h, c)
                ys = sample_from_dirichlet(alphas, n_clusters=self.n_clusters) 
                ys = ys / torch.sum(ys, -1).view(-1, 1)
                if self.discrete_decode:
                    mc_indices = torch.argmax(ys, dim=1)
                    ys[:, :] = 0
                    ys[torch.arange(len(ys)), mc_indices] = 1
                rep = self.G.sample(batch_size, x=sample(mu, log_var))[-1] + self.cluster_layer(ys)
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
            'G': self.G.state_dict(),
            'cluster_layer': self.cluster_layer.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])
        self.G.load_state_dict(checkpoint['G'])

        self.cluster_layer.load_state_dict(checkpoint['cluster_layer'])


class IAFDiffVAE(DiffVAE):
    def __init__(self, config):
        self.context_size = config.model.context_size
        self.p_weight = config.model.p_weight
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.iaf_layers = torch.nn.ModuleList([IAF(self.latent_size, self.context_size, parity=i % 2) for i in range(4)])

        if self.g_path is None:
            # Warm up
            self.G = self._train_diff()
        else:
            self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)
            checkpoint = torch.load(self.g_path)
            self.G.load_state_dict(checkpoint['Q'])
        self.G.eval()

        if self.endtoend:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.iaf_layers.parameters()) + list(self.G.parameters()), 
                                lr=self.lr)
        else:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.iaf_layers.parameters()), 
                                lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                        milestones=[150], 
                                                        gamma=0.1)

        self.Q.to(self.device)
        self.P.to(self.device)
        self.iaf_layers.to(self.device)
        self.G.to(self.device)
    
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
            self.G.train()
            prior_samples = self.prior.sample(self.batch_size)
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
                
                z = self.G.sample(batch_size, x=rep)[-1]
                inputs_reconstructed = self.P(z)

                reg_loss2 = torch.mean(log_det)
                reg_loss3 = self.prior.evaluate(rep)
                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = reg_loss1 - reg_loss2 - self.p_weight * reg_loss3

                if self.endtoend:
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    diff_loss = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")
                else:
                    diff_loss = torch.zeros(1).to(self.device)

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.diff_weight * diff_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), diff_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.G.eval()
        reps = []
        zs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, var_s, context = self.Q(inputs)

                var = torch.sigmoid(var_s)
                log_var = torch.log(var)

                rep = sample(mu, log_var)
                for layer in self.iaf_layers: 
                    rep, _ = layer(rep, context)
                z = self.G.sample(batch_size, x=rep)[-1]
            zs.append(z.cpu().numpy())
            reps.append(rep.cpu().numpy())
        return np.concatenate(zs, axis=0), np.concatenate(reps, axis=0)

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'G': self.G.state_dict(),
            'P': self.P.state_dict(),
            'iaf_layers': self.iaf_layers.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.G.load_state_dict(checkpoint['G'])
        self.P.load_state_dict(checkpoint['P'])
        self.iaf_layers.load_state_dict(checkpoint['iaf_layers'])

    def generate(self, n):
        all_outputs = []
        for i in range(0, n, self.batch_size):
            samples = self.prior.sample(min(self.batch_size, n-i))
            outputs = self.P(samples)
            all_outputs.append(outputs.cpu().detach().numpy())
        return np.concatenate(all_outputs, axis=0)
    

class DiffVAE_semi(DiffVAE):
    def __init__(self, config):
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_classes = config.dataset.n_classes
        self.cls_weight = config.model.cls_weight
        self.prior = AddNoise(get_prior(config), config.model.noise_sigma)
        self.priors = [PartitionPrior(self.prior, i) for i in range(self.n_classes)]
        super().__init__(config)
        if config.dataset.name.startswith('cifar'):
            self.kld_weights = get_kld_weights(0.5, 
                                               self.num_epochs, 
                                               config.model.kld_schedule, 
                                               config.model.kld_warmup)

    # TODO: Get G[i]
    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        if self.g_path is None:
            # Warm up
            self.G = self._train_diff()
        else:
            self.G = []
            for i in range(self.n_classes):
                one_g = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                            self.timesteps,
                                            self.latent_size,
                                            device=self.device)
                checkpoint = torch.load(os.path.join(self.g_path, f'diff_model{i}.pt'))
                one_g.load_state_dict(checkpoint['Q'])
                self.G.append(one_g)
            self.G = nn.ModuleList(self.G)
        self.G.eval()

        if self.endtoend:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.G.parameters()), 
                                lr=self.lr)
        else:
            self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), 
                                lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                        milestones=[150], 
                                                        gamma=0.1)

        self.Q.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)

    def _train_diff(self):
        G = []
        for i in range(self.n_classes):
            diff_config = DiffusionConfig
            diff_config.num_epochs = 100
            diff_config.prior = self.priors[i]
            diff_config.latent_size = self.latent_size
            diff_config.timesteps = self.timesteps
            diff_config.save_folder = os.path.join(self.save_folder, f'diff{i}')
            diff = StandardDiffusion(diff_config)
            diff.train()
            diff.save(os.path.join(self.save_folder, f'diff_model{i}.pt'))
            G.append(diff.Q)
        return nn.ModuleList(G)
    
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
            self.G.train()
            random_idx = np.random.randint(self.n_classes)
            prior_samples = self.priors[random_idx].sample(self.batch_size)
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var, cs_logits = self.Q(inputs)
                cs = F.softmax(cs_logits)
                x = sample(mu, log_var)

                unlabeled_indices = torch.arange(inputs.size(0))[targets_cpu == self.n_classes]
                unlabeled_loss = torch.zeros(1).to(self.device)
                labeled_loss = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    z = self.G[i].sample(batch_size, x=x)[-1]
                    inputs_reconstructed = self.P(z)
                    unlabeled_loss = unlabeled_loss + torch.sum(cs[unlabeled_indices, i].view(-1, *([1] * (len(inputs.size())-1))) * self.criterion_nored(inputs_reconstructed[unlabeled_indices], inputs[unlabeled_indices]))
                    labeled_loss = labeled_loss + torch.sum(self.criterion_nored(inputs_reconstructed[targets_cpu == i], inputs[targets_cpu == i]))
                rec_loss = (unlabeled_loss / len(unlabeled_indices) + labeled_loss / (max(1, inputs.size(0) - len(unlabeled_indices)))) / sum(inputs.size()[1:])
                
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                reg_loss = reg_loss + 0.001 * cs_entropy(cs[unlabeled_indices])

                if self.endtoend:
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    diff_loss = self.G[random_idx].loss_at_step_t(prior_samples, t, loss_type="huber")
                else:
                    diff_loss = torch.zeros(1).to(self.device)

                if len(targets[targets_cpu != self.n_classes]) == 0:
                    cls_loss = torch.zeros(1).to(self.device)
                else:
                    cls_loss = self.ce(cs_logits[targets_cpu != self.n_classes], targets[targets_cpu != self.n_classes])

                loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.diff_weight * diff_loss + self.cls_weight * self.kld_weights[self.ep] * cls_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), diff_loss.detach().item(), cls_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.G.eval()
        reps = []
        xs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var, cs_logits = self.Q(inputs)
                predicted_cs = torch.argmax(cs_logits, 1)
                x = sample(mu, log_var)
                rep = torch.zeros_like(x).to(self.device)
                for i in range(self.n_classes):
                    if len(x[predicted_cs == i]) > 0:
                        rep[predicted_cs == i] = self.G[i].sample(x[predicted_cs == i].size(0), x=x[predicted_cs == i])[-1]
            reps.append(rep.cpu().numpy())
            xs.append(x.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(xs, axis=0)
    
    def get_elbo(self, dataloader):
        self.Q.eval()
        self.G.eval()
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var, cs_logits = self.Q(inputs)
                cs = F.softmax(cs_logits)
                y = sample(mu, log_var)

                # \sum_{t=2}^T \KL (q(y_{t-1}|y_, x) || r(y_{t-1}|y_t, y_0)) where y_0 = z
                sm_kl = torch.zeros(1).to(self.device)
                prior_z = torch.zeros(1).to(self.device)
                rec = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    zs = self.G[i].sample(y.size(0), x=y, get_xt=True)
                    rep = zs[-1]
                    for tt in range(1, self.timesteps):
                        t = torch.full((batch_size,), tt, device=self.device, dtype=torch.long)
                        sm_kl = sm_kl + torch.sum(cs[:,i] * self.G[i].kl_divergence_at_t(zs[-1], zs[-1-(tt+1)], t, zs[-1-tt], nored=True))

                    prior_z = prior_z + torch.sum(cs[:,i] * -self.prior.evaluate_partition(rep, i))

                    inputs_reconstructed = self.P(rep)
                    # E_{q(y, z| x)} [log p(x|z)]
                    rec = rec + torch.sum(cs[:,i] * -torch.mean(self.criterion_nored(inputs_reconstructed, inputs).view(batch_size, -1), -1))

                sm_kl = sm_kl / batch_size
                prior_z = prior_z / batch_size
                rec = rec / batch_size

                # \KL (q(y_T|x) || r(y_T|y_0)) ==> \KL (q(y|x) || N(0, 1))
                prior_y = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            
            elbos.append(rec - prior_y - prior_z - sm_kl)
        log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()
    

class VAEDiffusion(DiffVAE):
    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                    self.timesteps,
                                    self.latent_size,
                                    device=self.device)
        # # TODO change back
        # self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
        #                                 self.timesteps,
        #                                 self.latent_size,
        #                                 device=self.device)

        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.G.parameters()), 
                            lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                        milestones=[150], 
                                                        gamma=0.1)

        self.Q.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)

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
            self.G.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var = self.Q(inputs)
                y = sample(mu, log_var)
                zs = self.G.sample(batch_size, x=y, get_xt=True)
                z = zs[-1]
                inputs_reconstructed = self.P(z)

                # E_{q(y, z| x)} [log p(x|z)]
                rec_loss = self.criterion(inputs_reconstructed, inputs)

                # \KL (q(y_T|x) || r(y_T|y_0)) ==> \KL (q(y|x) || N(0, 1))
                prior_y = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                # \KL (q(y_0|y_1, x) || p(y_0)) where y_0 = z
                variance_t0 = self.G.posterior_variance[0]
                # prior_z = torch.mean(gauss_logpdf_samevalue(torch.sqrt(variance_t0)) - self.prior.evaluate(z))
                prior_z = torch.mean(-self.prior.evaluate(z)) # variance = 0 so pdf = 1 and logpdf = 0


                # \sum_{t=2}^T \KL (q(y_{t-1}|y_, x) || r(y_{t-1}|y_t, y_0)) where y_0 = z
                sm_kl = torch.zeros(1).to(self.device)
                for tt in range(1, self.timesteps):
                    t = torch.full((batch_size,), tt, device=self.device, dtype=torch.long)
                    sm_kl = sm_kl + self.G.kl_divergence_at_t(zs[-1], zs[-1-(tt+1)], t, zs[-1-tt])
                reg_loss = prior_y + prior_z + sm_kl
                loss = rec_loss + self.kld_weights[self.ep] * reg_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def get_elbo(self, dataloader):
        self.Q.eval()
        self.G.eval()
        self.P.eval()
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var = self.Q(inputs)
                y = sample(mu, log_var)
                zs = self.G.sample(batch_size, x=y, get_xt=True)
                z = zs[-1]
                inputs_reconstructed = self.P(z)

                # E_{q(y, z| x)} [log p(x|z)]
                rec = -self.criterion(inputs_reconstructed, inputs)

                # \KL (q(y_T|x) || r(y_T|y_0)) ==> \KL (q(y|x) || N(0, 1))
                prior_y = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                # \KL (q(y_0|y_1, x) || p(y_0)) where y_0 = z
                prior_z = torch.mean(-self.prior.evaluate(z))


                # \sum_{t=2}^T \KL (q(y_{t-1}|y_, x) || r(y_{t-1}|y_t, y_0)) where y_0 = z
                sm_kl = torch.zeros(1).to(self.device)
                for tt in range(1, self.timesteps):
                    t = torch.full((batch_size,), tt, device=self.device, dtype=torch.long)
                    sm_kl = sm_kl + self.G.kl_divergence_at_t(zs[-1], zs[-1-(tt+1)], t, zs[-1-tt])
            
            elbos.append(rec - prior_y - prior_z - sm_kl)
        log.info([rec, prior_y, prior_z, sm_kl])
        return torch.mean(torch.stack(elbos)).cpu().item()


class DiffVAEFull(VAE):
    def __init__(self, config):
        self.g_path = config.g_path
        self.timesteps = config.model.timesteps
        self.prior = AddNoise(get_prior(config), config.model.noise_sigma)
        self.batch_size = config.model.batch_size
        self.endtoend = config.model.endtoend
        self.diff_weight = config.model.diff_weight
        self.context_size = config.model.context_size
        self.prior_z_weight = config.model.prior_z_weight
        self.diff_ct = config.model.diff_ct
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.G = GaussianDiffusionNoise(ConditionalModelWAUX(self.latent_size + self.context_size, self.timesteps, self.latent_size), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)

        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.G.parameters()), 
                            lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                        milestones=[150], 
                                                        gamma=0.1)

        self.Q.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)
    
    def _call_schedulers(self):
        self.scheduler.step()
        
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
            self.G.train()
            

            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var, _ = self.Q(inputs)
                aux = torch.zeros((batch_size, self.context_size)).to(self.device)
                x = sample(mu, log_var)
                # z = self.G.sample(batch_size, x=x, aux=aux)[-1]
                zs = self.G.sample(batch_size, x=x, aux=aux, get_xt=True)
                z = zs[-1]
                inputs_reconstructed = self.P(z)

                # TODO Consider removing this
                stacked_zs = torch.stack(zs)
                all_diff_loss = torch.zeros(1).to(self.device)
                for _ in range(self.diff_ct):
                    t = torch.randint(
                        0, self.timesteps, (batch_size,), device=self.device
                    ).long()
                    diff_loss = self.G.weird_loss_at_step_t(stacked_zs[-1], stacked_zs[-1-(t+1), torch.arange(batch_size)], t, aux, loss_type="huber")
                    all_diff_loss = all_diff_loss + diff_loss

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                reg_loss2 = -self.prior.evaluate(z)
                loss = rec_loss + self.kld_weights[self.ep] * (reg_loss + self.prior_z_weight * reg_loss2) + self.diff_weight * all_diff_loss
                loss.backward()
                self.optim.step()

            for _ in range(self.diff_ct):
                prior_samples = self.prior.sample(self.batch_size)
                with torch.set_grad_enabled(False):
                    fantasy_inputs = self.P(prior_samples)

                self.optim.zero_grad()
                with torch.set_grad_enabled(True):
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    _, _, fantasy_aux = self.Q(fantasy_inputs)
                    diff_loss = self.G.loss_at_step_t(prior_samples, t, aux=fantasy_aux, loss_type="huber")

                    loss = self.diff_weight * diff_loss
                    loss.backward()
                    self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), diff_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.G.eval()
        reps = []
        xs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var, aux = self.Q(inputs)
                x = sample(mu, log_var)
                rep = self.G.sample(batch_size, x=x, aux=aux)[-1]
            reps.append(rep.cpu().numpy())
            xs.append(x.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(xs, axis=0)

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'G': self.G.state_dict(),
            'P': self.P.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.G.load_state_dict(checkpoint['G'])
        self.P.load_state_dict(checkpoint['P'])


class DiffVAEBoth(VAE):
    def __init__(self, config):
        self.g_path = config.g_path
        self.timesteps = config.model.timesteps
        self.prior = AddNoise(get_prior(config), config.model.noise_sigma)
        self.batch_size = config.model.batch_size
        self.endtoend = config.model.endtoend
        self.diff_weight = config.model.diff_weight
        self.diff_weight2 = config.model.diff_weight2
        self.prior_z_weight = config.model.prior_z_weight
        self.context_size = config.model.context_size
        self.diff_ct = config.model.diff_ct
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.G = self._train_diff()
        # self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
        #                                 self.timesteps,
        #                                 self.latent_size,
        #                                 device=self.device)
        # checkpoint = torch.load(os.path.join(self.save_folder, 'diff_model.pt'))
        # self.G.load_state_dict(checkpoint['Q'])
        
        self.G2 = GaussianDiffusionNoise(ConditionalModelWAUX(self.latent_size + self.context_size, self.timesteps, self.latent_size), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)
        
        self.doubleG = TwoDiffusion(self.G, self.G2, self.timesteps, self.latent_size, device=self.device)

        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.G.parameters()) + list(self.G2.parameters()), 
                            lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                                                        milestones=[150], 
                                                        gamma=0.1)

        self.Q.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)
        self.G2.to(self.device)

    def _train_diff(self):
        diff_config = DiffusionConfig
        diff_config.num_epochs = 100
        diff_config.prior = self.prior
        diff_config.latent_size = self.latent_size
        diff_config.timesteps = self.timesteps
        diff_config.save_folder = os.path.join(self.save_folder, 'diff')
        diff = StandardDiffusion(diff_config)
        diff.train()
        diff.save(os.path.join(self.save_folder, 'diff_model.pt'))
        return diff.Q
    
    def _call_schedulers(self):
        self.scheduler.step()
        
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
            self.G.train()
            self.G2.train()

            prior_samples = self.prior.sample(self.batch_size)
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var, _ = self.Q(inputs)
                aux = torch.zeros((batch_size, self.context_size)).to(self.device)
                x = sample(mu, log_var)
                # z = self.G.sample(batch_size, x=x, aux=aux)[-1]
                zs = self.doubleG.sample(batch_size, x=x, aux=aux, get_xt=True)
                z = zs[-1]
                inputs_reconstructed = self.P(z)

                t = torch.randint(
                    0, self.timesteps, (self.batch_size,), device=self.device
                ).long()
                diff_loss_v1 = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")

                # TODO Consider removing this
                stacked_zs = torch.stack(zs)
                all_diff_loss_v2 = torch.zeros(1).to(self.device)
                for _ in range(self.diff_ct):
                    t = torch.randint(
                        0, self.timesteps, (batch_size,), device=self.device
                    ).long()
                    diff_loss_v2 = self.G2.weird_loss_at_step_t(stacked_zs[-1], stacked_zs[-1-(t+1), torch.arange(batch_size)], t, aux, loss_type="huber")
                    all_diff_loss_v2 = all_diff_loss_v2 + diff_loss_v2

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                reg_loss2 = -self.prior.evaluate(z) # TODO Should we remove this
                loss = rec_loss + self.kld_weights[self.ep] * (reg_loss + self.prior_z_weight * reg_loss2) + self.diff_weight * diff_loss_v1 + self.diff_weight2 * all_diff_loss_v2
                loss.backward()
                self.optim.step()

            diff_loss = torch.zeros(1).to(self.device)
            for _ in range(self.diff_ct):
                prior_samples = self.prior.sample(self.batch_size)
                with torch.set_grad_enabled(False):
                    fantasy_inputs = self.P(prior_samples)

                self.optim.zero_grad()
                with torch.set_grad_enabled(True):
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    _, _, fantasy_aux = self.Q(fantasy_inputs)
                    diff_loss = self.G2.loss_at_step_t(prior_samples, t, aux=fantasy_aux, loss_type="huber")

                    loss = self.diff_weight2 * diff_loss
                    loss.backward()
                    self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), diff_loss_v1.detach().item(), diff_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.G.eval()
        reps = []
        xs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var, aux = self.Q(inputs)
                x = sample(mu, log_var)
                rep = self.doubleG.sample(batch_size, x=x, aux=aux)[-1]
            reps.append(rep.cpu().numpy())
            xs.append(x.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(xs, axis=0)

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'G': self.G.state_dict(),
            'G2': self.G2.state_dict(),
            'P': self.P.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.G.load_state_dict(checkpoint['G'])
        self.G2.load_state_dict(checkpoint['G2'])
        self.P.load_state_dict(checkpoint['P'])


class DiffVAEWarmup(VAE):
    def __init__(self, config):
        self.g_path = config.g_path
        self.timesteps = config.model.timesteps
        self.prior = AddNoise(get_prior(config), config.model.noise_sigma)
        self.batch_size = config.model.batch_size
        self.endtoend = config.model.endtoend
        self.diff_weight = config.model.diff_weight
        self.diff_weight2 = config.model.diff_weight2
        self.prior_z_weight = config.model.prior_z_weight
        self.context_size = config.model.context_size
        self.diff_ct = config.model.diff_ct
        self.dropout = config.model.dropout if config.dataset.name.startswith('mnist') else 0.2
        self.g2_weight = config.model.g2_weight
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        if self.config.dataset.name.startswith('mnist'):
            self.Q2 = MultiWayLinear(784, [self.config.model.hidden_size, self.config.model.hidden_size], self.config.model.context_size, bn=self.config.model.bn, dropout=self.dropout)
        else:
            self.Q2 = nn.Sequential(BareConvNet(), 
                                    MultiWayLinear(128, [24, 24],  self.config.model.context_size, bn=self.config.model.bn, dropout=self.dropout))

        # self.G = self._train_diff()
        g_timesteps = 20
        self.G = GaussianDiffusionNoise(ConditionalModel(self.latent_size, g_timesteps), 
                                        g_timesteps,
                                        self.latent_size,
                                        device=self.device)
        # checkpoint = torch.load(os.path.join('experimental_results/mnist_diff_vae_warmup_less_noisy_square_contextsize2_diffct5_kldw0.003', 'diff_model.pt'))
        # self.G.load_state_dict(checkpoint['Q'])
        # if self.config.prior.type in ['swiss_roll', 'pin_wheel']:
        #     checkpoint = torch.load(os.path.join('experimental_results', f'new_mnist_diff_vae_{self.config.prior.type}_kldw0.01', 'model.pt'))
        # elif self.config.prior.type == 'square':
        #     name = 'mnist_diff_vae_less_noisy_square'
        #     checkpoint = torch.load(os.path.join('final_results', name, 'run2', 'model.pt'))
        # else:
        #     raise NotImplementedError
        if self.config.prior.type == 'square':
            prior_name = 'less_noisy_square'
        elif self.config.prior.type == 'pin_wheel':
            prior_name = 'four_pin_wheel' if self.config.prior.n_arc == 4 else 'pin_wheel'
        else:
            prior_name = self.config.prior.type
        checkpoint = torch.load(os.path.join(self.save_folder.split('/')[0], f'{self.config.dataset.name}_diff_vae_{prior_name}', f'run{self.config.seed}', 'model.pt'))
        # checkpoint = torch.load(os.path.join('experimental_results', 'supernew_mnist_diff_vae_pin_wheel', 'model.pt')) # TODO change back!
        self.Q.load_state_dict(checkpoint['Q'])
        self.G.load_state_dict(checkpoint['G'])
        self.P.load_state_dict(checkpoint['P'])

        
        self.G2 = GaussianDiffusionNoise(ConditionalModelWAUX(self.latent_size + self.context_size, self.timesteps, self.latent_size), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)
        # self.G2 = GaussianDiffusionNoise(StackedMADE(self.latent_size, self.timesteps, self.latent_size), 
        #                                 self.timesteps,
        #                                 self.latent_size,
        #                                 device=self.device)
        
        self.doubleG = TwoDiffusion(self.G, self.G2, self.timesteps, self.latent_size, device=self.device, g2_weight=self.g2_weight)

        self.middle_optim = optim.Adam(list(self.Q.parameters()) + list(self.Q2.parameters())+ list(self.G2.parameters()), 
                            lr=self.lr)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.Q2.parameters()) + list(self.G2.parameters()), 
                            lr=self.lr)
        self.scheduler = None

        self.Q.to(self.device)
        self.Q2.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)
        self.G2.to(self.device)

    def _train_diff(self):
        diff_config = DiffusionConfig
        diff_config.num_epochs = 100
        diff_config.prior = self.prior
        diff_config.latent_size = self.latent_size
        diff_config.timesteps = self.timesteps
        diff_config.save_folder = os.path.join(self.save_folder, 'diff')
        diff = StandardDiffusion(diff_config)
        diff.train()
        diff.save(os.path.join(self.save_folder, 'diff_model.pt'))
        return diff.Q
    
    def _call_schedulers(self):
        pass

    # def _train_epoch_firststage(self, dataloader):
    #     '''
    #     Train procedure for one epoch.
    #     '''

    #     all_res = 0
    #     for inputs in dataloader:
    #         inputs = inputs.to(self.device)

    #         batch_size = inputs.size()[0]
            
    #         self.P.train()
    #         self.Q.train()
    #         self.G.train()
    #         prior_samples = self.prior.sample(self.batch_size)
    #         self.optim.zero_grad()
    #         with torch.set_grad_enabled(True):
    #             mu, log_var, _ = self.Q(inputs)
    #             x = sample(mu, log_var)
    #             z = self.G.sample(batch_size, x=x)[-1]
    #             inputs_reconstructed = self.P(z)

    #             rec_loss = self.criterion(inputs_reconstructed, inputs)
    #             reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    #             if self.endtoend:
    #                 t = torch.randint(
    #                     0, self.timesteps, (self.batch_size,), device=self.device
    #                 ).long()
    #                 diff_loss = self.G.loss_at_step_t(prior_samples, t, loss_type="huber")
    #             else:
    #                 diff_loss = torch.zeros(1).to(self.device)

    #             loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.diff_weight * diff_loss
    #             loss.backward()
    #             self.optim.step()
    #         res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), diff_loss.detach().item()])
    #         all_res += res
        
    #     l = len(dataloader)
    #     return all_res / l
        
    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''
        # if self.ep < 50:
        #     return self._train_epoch_firststage(dataloader)
        
        # if self.ep == 50:
        #     self.save(os.path.join(self.save_folder, 'v1model.pt'))

        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]
            
            self.P.train()
            self.Q.train()
            self.Q2.train()
            self.G.train()
            self.G2.train()
            
            if self.ep < 30:
                self.middle_optim.zero_grad()
            else:
                self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var = self.Q(inputs)
                aux = self.Q2(inputs)
                x = sample(mu, log_var)
                zs = self.doubleG.sample(batch_size, x=x, aux=aux, get_xt=True)
                z = zs[-1]
                inputs_reconstructed = self.P(z)

                stacked_zs = torch.stack(zs)
                all_diff_loss = torch.zeros(1).to(self.device)
                # for tt in range(0, self.timesteps):
                #     t = torch.full((batch_size,), tt, device=self.device, dtype=torch.long)
                t = torch.randint(
                    0, self.timesteps, (batch_size,), device=self.device
                ).long()
                # t = torch.full((batch_size,), np.random.randint(self.timesteps), device=self.device, dtype=torch.long) # TODO change back
                diff_loss = self.G2.weird_loss_at_step_t(stacked_zs[-1], stacked_zs[-1-(t+1), torch.arange(batch_size)], t, aux, loss_type="huber")
                all_diff_loss = all_diff_loss + diff_loss

                rec_loss = self.criterion(inputs_reconstructed, inputs)
                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                reg_loss2 = -self.prior.evaluate(z)
                loss = rec_loss + self.kld_weights[self.ep] * (reg_loss + self.prior_z_weight * reg_loss2) + self.diff_weight * all_diff_loss
                loss.backward()
                if self.ep < 30:
                    self.middle_optim.step()
                else:
                    self.optim.step()

            for _ in range(self.diff_ct):
                prior_samples = self.prior.sample(self.batch_size)
                with torch.set_grad_enabled(False):
                    fantasy_inputs = self.P(prior_samples)

                self.optim.zero_grad()
                with torch.set_grad_enabled(True):
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    # t = torch.full((self.batch_size,), np.random.randint(self.timesteps), device=self.device, dtype=torch.long) # TODO change back
                    fantasy_aux = self.Q2(fantasy_inputs)
                    diff_loss = self.G2.loss_at_step_t(prior_samples, t, aux=fantasy_aux, loss_type="huber")

                    loss = self.diff_weight * diff_loss
                    loss.backward()
                    self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), reg_loss2.detach().item(), all_diff_loss.detach().item(), diff_loss.detach().item(), (rec_loss + reg_loss + reg_loss2 + all_diff_loss).detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l
    
    def eval(self, dataloader):
        self.Q.eval()
        self.G2.eval()
        reps = []
        xs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var = self.Q(inputs)
                aux = self.Q2(inputs)
                x = sample(mu, log_var)
                rep = self.doubleG.sample(batch_size, x=x, aux=aux)[-1]
            reps.append(rep.cpu().numpy())
            xs.append(x.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(xs, axis=0)
    
    def get_elbo(self, dataloader):
        self.Q.eval()
        self.Q2.eval()
        self.G.eval()
        self.G2.eval()
        self.P.eval()
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var = self.Q(inputs)
                aux = self.Q2(inputs)
                y = sample(mu, log_var)
                zs = self.doubleG.sample(batch_size, x=y, aux=aux, get_xt=True)
                z = zs[-1]
                inputs_reconstructed = self.P(z)

                # E_{q(y, z| x)} [log p(x|z)]
                rec = -self.criterion(inputs_reconstructed, inputs)

                # \KL (q(y_T|x) || r(y_T|y_0)) ==> \KL (q(y|x) || N(0, 1))
                prior_y = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                # \KL (q(y_0|y_1, x) || p(y_0)) where y_0 = z
                variance_t0 = self.doubleG.posterior_variance[0]
                # prior_z = torch.mean(gauss_logpdf_samevalue(torch.sqrt(variance_t0)) - self.prior.evaluate(z))
                prior_z = torch.mean(-self.prior.evaluate(z)) # variance = 0 so pdf = 1 and logpdf = 0


                # \sum_{t=2}^T \KL (q(y_{t-1}|y_, x) || r(y_{t-1}|y_t, y_0)) where y_0 = z
                sm_kl = torch.zeros(1).to(self.device)
                for tt in range(1, self.timesteps):
                    t = torch.full((batch_size,), tt, device=self.device, dtype=torch.long)
                    sm_kl = sm_kl + self.doubleG.kl_divergence_at_t(zs[-1], zs[-1-(tt+1)], t, zs[-1-tt])
            
            elbos.append(rec - prior_y - prior_z - sm_kl)
        return torch.mean(torch.stack(elbos)).cpu().item()


    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'Q2': self.Q2.state_dict(),
            'G': self.G.state_dict(),
            'G2': self.G2.state_dict(),
            'P': self.P.state_dict(),
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.Q2.load_state_dict(checkpoint['Q2'])
        self.G.load_state_dict(checkpoint['G'])
        self.G2.load_state_dict(checkpoint['G2'])
        self.P.load_state_dict(checkpoint['P'])


class DiffVAEWarmup_semi(DiffVAEWarmup):
    def __init__(self, config):
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_classes = config.dataset.n_classes
        self.cls_weight = config.model.cls_weight
        self.prior = AddNoise(get_prior(config), config.model.noise_sigma)
        self.priors = [PartitionPrior(self.prior, i) for i in range(self.n_classes)]
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        if self.config.dataset.name.startswith('mnist'):
            self.Q2 = MultiWayLinear(784, [self.config.model.hidden_size, self.config.model.hidden_size], self.config.model.context_size, bn=self.config.model.bn, dropout=self.dropout)
        else:
            self.Q2 = nn.Sequential(BareConvNet(), 
                                    MultiWayLinear(128, [24, 24],  self.config.model.context_size, bn=self.config.model.bn, dropout=self.dropout))

        # self.G = self._train_diff()
        g_timesteps = 20
        self.G = []
        for i in range(self.n_classes):
            one_g = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                        g_timesteps,
                                        self.latent_size,
                                        device=self.device)
            self.G.append(one_g)
        self.G = nn.ModuleList(self.G)


        if self.config.prior.type == 'square':
            prior_name = 'less_noisy_square'
        elif self.config.prior.type == 'pin_wheel':
            prior_name = 'four_pin_wheel' if self.config.prior.n_arc == 4 else 'pin_wheel'
        else:
            prior_name = 'swiss_roll'
        if self.config.dataset.name.startswith('mnist'):
            folder = 'new_extrasemi_final_results' if self.config.dataset.n_labels != 1000 else 'new_final_results'
            n_label = f'{self.config.dataset.n_labels}' if self.config.dataset.n_labels != 1000 else ''
        else:
            folder = 'new_extrasemi_final_results' if self.config.dataset.n_labels != 1000 else 'new_final_results'
            n_label = f'{self.config.dataset.n_labels}' if self.config.dataset.n_labels != 10000 else ''
        checkpoint = torch.load(os.path.join(folder, f'{self.config.dataset.name}{n_label}_diff_vae_semi_{prior_name}', f'run{self.config.seed}', 'model.pt'))
        self.Q.load_state_dict(checkpoint['Q'])
        self.G.load_state_dict(checkpoint['G'])
        self.P.load_state_dict(checkpoint['P'])

        self.G2 = []
        for i in range(self.n_classes):
            one_g = GaussianDiffusionNoise(ConditionalModelWAUX(self.latent_size + self.context_size, self.timesteps, self.latent_size), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=self.device)
            self.G2.append(one_g)
        self.G2 = nn.ModuleList(self.G2)
        
        self.doubleG = TwoDiffusion(self.G, self.G2, self.timesteps, self.latent_size, device=self.device, g2_weight=self.g2_weight)

        self.middle_optim = optim.Adam(list(self.Q.parameters()) + list(self.Q2.parameters())+ list(self.G2.parameters()), 
                            lr=self.lr)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.Q2.parameters()) + list(self.G2.parameters()), 
                            lr=self.lr)
        self.scheduler = None

        self.Q.to(self.device)
        self.Q2.to(self.device)
        self.P.to(self.device)
        self.G.to(self.device)
        self.G2.to(self.device)

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''
        # if self.ep < 50:
        #     return self._train_epoch_firststage(dataloader)
        
        # if self.ep == 50:
        #     self.save(os.path.join(self.save_folder, 'v1model.pt'))

        all_res = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets_cpu = targets
            targets = targets.to(self.device)

            batch_size = inputs.size()[0]
            
            self.P.train()
            self.Q.train()
            self.Q2.train()
            self.G.train()
            self.G2.train()
            
            if self.ep < 30:
                self.middle_optim.zero_grad()
            else:
                self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                mu, log_var, cs_logits = self.Q(inputs)
                aux = self.Q2(inputs)
                cs = F.softmax(cs_logits)
                x = sample(mu, log_var)
                
                unlabeled_indices = torch.arange(inputs.size(0))[targets_cpu == self.n_classes]
                unlabeled_loss = torch.zeros(1).to(self.device)
                labeled_loss = torch.zeros(1).to(self.device)
                unlabeled_prior_loss = torch.zeros(1).to(self.device)
                labeled_prior_loss = torch.zeros(1).to(self.device)
                unlabeled_diff_loss = torch.zeros(1).to(self.device)
                labeled_diff_loss = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    zs = self.G2[i].sample(batch_size, x=x, aux=aux, get_xt=True)
                    z = zs[-1]
                    inputs_reconstructed = self.P(z)
                    unlabeled_loss = unlabeled_loss + torch.sum(cs[unlabeled_indices, i].view(-1, *([1] * (len(inputs.size())-1))) * self.criterion_nored(inputs_reconstructed[unlabeled_indices], inputs[unlabeled_indices]))
                    labeled_loss = labeled_loss + torch.sum(self.criterion_nored(inputs_reconstructed[targets_cpu == i], inputs[targets_cpu == i]))

                    unlabeled_prior_loss = unlabeled_prior_loss + torch.sum(cs[unlabeled_indices, i].view(-1) * (-self.prior.evaluate(z[unlabeled_indices], nored=True)))
                    labeled_prior_loss = labeled_prior_loss + torch.sum((-self.prior.evaluate(z[targets_cpu == i], nored=True)))

                    stacked_zs = torch.stack(zs)
                    t = torch.randint(
                        0, self.timesteps, (batch_size,), device=self.device
                    ).long()
                    unlabeled_diff_loss = unlabeled_diff_loss + torch.sum(cs[unlabeled_indices, i].view(-1) * torch.mean(self.G2[i].weird_loss_at_step_t((stacked_zs[-1])[unlabeled_indices], (stacked_zs[-1-(t+1), torch.arange(batch_size)])[unlabeled_indices], t[unlabeled_indices], aux[unlabeled_indices], loss_type="huber", nored=True), -1))
                    labeled_diff_loss = labeled_diff_loss + torch.sum(torch.mean(self.G2[i].weird_loss_at_step_t((stacked_zs[-1])[targets_cpu == i], (stacked_zs[-1-(t+1), torch.arange(batch_size)])[targets_cpu == i], t[targets_cpu == i], aux[targets_cpu == i], loss_type="huber", nored=True), -1))

                rec_loss = (unlabeled_loss / len(unlabeled_indices) + labeled_loss / (max(1, inputs.size(0) - len(unlabeled_indices)))) / sum(inputs.size()[1:])

                reg_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                reg_loss = reg_loss + 0.001 * cs_entropy(cs[unlabeled_indices])

                reg_loss2 = (unlabeled_prior_loss / len(unlabeled_indices) + labeled_prior_loss / (max(1, inputs.size(0) - len(unlabeled_indices))))

                all_diff_loss = (unlabeled_diff_loss / len(unlabeled_indices) + labeled_diff_loss / (max(1, inputs.size(0) - len(unlabeled_indices))))

                loss = rec_loss + self.kld_weights[self.ep] * (reg_loss + self.prior_z_weight * reg_loss2) + self.diff_weight * all_diff_loss
                loss.backward()
                if self.ep < 30:
                    self.middle_optim.step()
                else:
                    self.optim.step()

            for _ in range(self.diff_ct):
                random_idx = np.random.randint(self.n_classes)
                prior_samples = self.priors[random_idx].sample(self.batch_size)
                with torch.set_grad_enabled(False):
                    fantasy_inputs = self.P(prior_samples)

                self.optim.zero_grad()
                with torch.set_grad_enabled(True):
                    t = torch.randint(
                        0, self.timesteps, (self.batch_size,), device=self.device
                    ).long()
                    # t = torch.full((self.batch_size,), np.random.randint(self.timesteps), device=self.device, dtype=torch.long) # TODO change back
                    fantasy_aux = self.Q2(fantasy_inputs)
                    diff_loss = self.G2[random_idx].loss_at_step_t(prior_samples, t, aux=fantasy_aux, loss_type="huber")

                    loss = self.diff_weight * diff_loss
                    loss.backward()
                    self.optim.step()
            res = np.asarray([rec_loss.detach().item(), reg_loss.detach().item(), reg_loss2.detach().item(), diff_loss.detach().item()])
            all_res += res
        
        l = len(dataloader)
        return all_res / l

    def eval(self, dataloader):
        self.Q.eval()
        self.Q2.eval()
        self.G.eval()
        self.G2.eval()
        self.P.eval()
        reps = []
        xs = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)

            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var, cs_logits = self.Q(inputs)
                aux = self.Q2(inputs)
                predicted_cs = torch.argmax(cs_logits, 1)
                x = sample(mu, log_var)
                rep = torch.zeros_like(x).to(self.device)
                for i in range(self.n_classes):
                    if len(x[predicted_cs == i]) > 0:
                        rep[predicted_cs == i] = self.G2[i].sample(x[predicted_cs == i].size(0), x=x[predicted_cs == i], aux=aux[predicted_cs == i])[-1]
            reps.append(rep.cpu().numpy())
            xs.append(x.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(xs, axis=0)
    
    def get_elbo(self, dataloader):
        self.Q.eval()
        self.Q2.eval()
        self.G.eval()
        self.G2.eval()
        self.P.eval()
        elbos = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            batch_size = inputs.size()[0]

            with torch.set_grad_enabled(False):
                mu, log_var, cs_logits = self.Q(inputs)
                cs = F.softmax(cs_logits)
                aux = self.Q2(inputs)
                y = sample(mu, log_var)

                # \sum_{t=2}^T \KL (q(y_{t-1}|y_, x) || r(y_{t-1}|y_t, y_0)) where y_0 = z
                sm_kl = torch.zeros(1).to(self.device)
                prior_z = torch.zeros(1).to(self.device)
                rec = torch.zeros(1).to(self.device)
                for i in range(self.n_classes):
                    zs = self.G2[i].sample(y.size(0), x=y, aux=aux, get_xt=True)
                    rep = zs[-1]
                    sub_batch_size = zs[-1].size()[0]
                    for tt in range(1, self.timesteps):
                        t = torch.full((sub_batch_size,), tt, device=self.device, dtype=torch.long)
                        sm_kl = sm_kl + sub_batch_size * self.G2[i].kl_divergence_at_t(zs[-1], zs[-1-(tt+1)], t, zs[-1-tt])
                
                    prior_z = prior_z + torch.sum(cs[:,i] * -self.prior.evaluate_partition(rep, i))

                    inputs_reconstructed = self.P(rep)
                    # E_{q(y, z| x)} [log p(x|z)]
                    rec = rec + torch.sum(cs[:,i] * -torch.mean(self.criterion_nored(inputs_reconstructed, inputs).view(batch_size, -1), -1))

                sm_kl = sm_kl / batch_size
                prior_z = prior_z / batch_size
                rec = rec / batch_size

                # \KL (q(y_T|x) || r(y_T|y_0)) ==> \KL (q(y|x) || N(0, 1))
                prior_y = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            
            elbos.append(rec - prior_y - prior_z - sm_kl)
        return torch.mean(torch.stack(elbos)).cpu().item()