"""Ref: https://github.com/fducau/AAE_pytorch"""
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .aae_vanilla import AAE_vanilla
from priors import sample_categorical
from .utils import GaussianNoise
from ..utils import get_q_and_p


log = logging.getLogger(__name__)


class D_net(nn.Module):
    def __init__(self, input_size, hidden_size=1000):
        super(D_net, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.noise = GaussianNoise()

    def forward(self, x):
        x = self.hidden_layer_1(self.noise(x))
        x = F.dropout(self.leaky_relu(x), 0.2)
        x = self.hidden_layer_2(x)
        x = F.dropout(self.leaky_relu(x), 0.2)
        x = self.output_layer(x)

        return F.sigmoid(x)


class AAE_w_cluster_heads(AAE_vanilla):
    def __init__(self, config):
        self.n_clusters = config.model.n_clusters
        self.eta = config.model.eta
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.cluster_layer = nn.Linear(self.n_clusters, self.latent_size, bias=False)
        self.D_cat = D_net(self.n_clusters)
        self.D_gauss = D_net(self.latent_size)
        
        self.Q.to(self.device)
        self.P.to(self.device)
        self.D_gauss.to(self.device)
        self.D_cat.to(self.device)
        self.cluster_layer.to(self.device)

        # Set optimizators
        self.ae_optim = optim.Adam(list(self.P.parameters()) + list(self.cluster_layer.parameters()) + list(self.Q.parameters()), 
                                       lr=0.0002)
        self.ae_scheduler = optim.lr_scheduler.MultiStepLR(self.ae_optim, 
                                                           milestones=[150], 
                                                           gamma=0.1)

        self.disc_optim = optim.Adam(list(self.D_cat.parameters()) + list(self.D_gauss.parameters()), lr=0.0002)
        self.disc_scheduler = optim.lr_scheduler.MultiStepLR(self.disc_optim, 
                                                             milestones=[150], 
                                                             gamma=0.1)
        
        self.gen_optim = optim.Adam(self.Q.parameters(), lr=0.0002)
        self.gen_scheduler = optim.lr_scheduler.MultiStepLR(self.gen_optim, 
                                                            milestones=[150], 
                                                            gamma=0.1)
        
    def _call_schedulers(self):
        self.ae_scheduler.step()
        self.disc_scheduler.step()
        self.gen_scheduler.step()
        self.D_gauss.noise.sigma = 0.1 * ((self.num_epochs - (self.ep + 1)) // (self.num_epochs // 4))
        self.D_cat.noise.sigma = 0.1 * ((self.num_epochs - (self.ep + 1)) // (self.num_epochs // 4))

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

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''
        EPS = 1e-6

        total_recon_loss, total_cluster_loss, total_D_cat_loss, total_D_gauss_loss, total_G_loss = 0, 0, 0, 0, 0
        for inputs in dataloader: # TODO: check how long this is
            batch_size = inputs.size(0)

            inputs = inputs.to(self.device)
            
            #######################
            # Reconstruction phase
            #######################
            self.Q.train()
            self.P.train()
            self.ae_optim.zero_grad()
            with torch.set_grad_enabled(True):
                logits, zs = self.Q(inputs)
                ys = F.softmax(logits)

                inputs_reconstructed = self.P(self.cluster_layer(ys) + zs)

                recon_loss = 0.5 * self.criterion(inputs_reconstructed, inputs)

                recon_loss.backward()
                self.ae_optim.step()
            total_recon_loss += recon_loss.detach().item()

            #######################
            # Regularization phase
            #######################
            ys_real = sample_categorical(batch_size, self.n_clusters)
            zs_real = self.prior.sample(batch_size)

            ys_real = ys_real.to(self.device)
            zs_real = zs_real.to(self.device)

            # Discriminator
            self.Q.eval()
            self.D_cat.train()
            self.D_gauss.train()
            self.disc_optim.zero_grad()
            with torch.set_grad_enabled(True):
                logits, zs = self.Q(inputs)
                ys = F.softmax(logits)

                D_real_cat = self.D_cat(ys_real)
                D_real_gauss = self.D_gauss(zs_real)
                D_fake_cat = self.D_cat(ys)
                D_fake_gauss = self.D_gauss(zs)

                D_loss_cat = -6 * torch.mean(torch.log(D_real_cat + EPS) + torch.log(1 - D_fake_cat + EPS))
                D_loss_gauss = -6 * torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

                D_loss = D_loss_cat + D_loss_gauss

                D_loss.backward()
                self.disc_optim.step()
            total_D_cat_loss += D_loss_cat.detach().item()
            total_D_gauss_loss += D_loss_gauss.detach().item()

            # Generator
            self.Q.train()
            self.D_cat.eval()
            self.D_gauss.eval()
            self.gen_optim.zero_grad()
            with torch.set_grad_enabled(True):
                logits, zs = self.Q(inputs)
                ys = F.softmax(logits)

                D_fake_cat = self.D_cat(ys)
                D_fake_gauss = self.D_gauss(zs)

                G_loss = -6 * torch.mean(torch.log(D_fake_cat + EPS) + torch.log(D_fake_gauss + EPS))
                G_loss.backward()
                self.gen_optim.step()
            total_G_loss += G_loss.detach().item()

            self.ae_optim.zero_grad()
            a_heads = sample_categorical(self.n_clusters, self.n_clusters)
            b_heads = sample_categorical(self.n_clusters, self.n_clusters)
            a_heads = a_heads.to(self.device)
            b_heads = b_heads.to(self.device)
            with torch.set_grad_enabled(True):
                cluster_loss = self._calc_cluster_loss(a_heads, b_heads)
                cluster_loss.backward()
                self.ae_optim.step()
            total_cluster_loss += cluster_loss.detach().item()

        l = len(dataloader)
        return total_recon_loss / l, total_cluster_loss / l, total_D_cat_loss / l, total_D_gauss_loss / l, total_G_loss / l

    def eval(self, dataloader):
        self.Q.eval()
        reps = []
        all_ys = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                logits, zs = self.Q(inputs)
                ys = F.softmax(logits)
                rep = self.cluster_layer(ys) + zs
            reps.append(rep.cpu().detach().numpy())
            all_ys.append(ys.cpu().numpy())
        return np.concatenate(reps, axis=0), np.concatenate(all_ys, axis=0)
    
    def eval_label(self, dataloader):
        _, ys = self.eval(dataloader)
        return torch.argmax(torch.tensor(ys), -1).cpu().numpy()

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            'D_cat': self.D_cat.state_dict(),
            'D_gauss': self.D_gauss.state_dict(),
            'cluster_layer': self.cluster_layer.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])
        self.D_cat.load_state_dict(checkpoint['D_cat'])
        self.D_gauss.load_state_dict(checkpoint['D_gauss'])
        self.cluster_layer.load_state_dict(checkpoint['cluster_layer'])
