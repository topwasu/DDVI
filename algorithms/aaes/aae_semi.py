"""Ref: https://github.com/fducau/AAE_pytorch"""
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .aae_vanilla import AAE_vanilla
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


class AAE_semi(AAE_vanilla):
    def __init__(self, config):
        self.n_classes = config.dataset.n_classes
        super().__init__(config)

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.D_gauss = D_net(self.latent_size + self.n_classes + 1) 
        
        self.Q.to(self.device)
        self.P.to(self.device)
        self.D_gauss.to(self.device)

        # Set optimizators
        self.ae_optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), 
                                       lr=0.0002)
        self.ae_scheduler = optim.lr_scheduler.MultiStepLR(self.ae_optim, 
                                                           milestones=[150], 
                                                           gamma=0.1)

        self.disc_optim = optim.Adam(self.D_gauss.parameters(), lr=0.0002)
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
        
    def _random_label_flip(self, labels):
        labels[np.random.choice(len(labels), int(0.5*len(labels)), replace=False)] = self.n_classes
        return labels

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''
        EPS = 1e-15

        total_recon_loss, total_D_loss, total_G_loss = 0, 0, 0
        for inputs, labels in dataloader: # TODO: check how long this is
            batch_size = inputs.size(0)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            #######################
            # Reconstruction phase
            #######################
            self.Q.train()
            self.P.train()
            self.ae_optim.zero_grad()
            with torch.set_grad_enabled(True):
                zs = self.Q(inputs)
                inputs_reconstructed = self.P(zs)

                recon_loss = 0.5 * self.criterion(inputs_reconstructed, inputs)
                recon_loss.backward()
                self.ae_optim.step()
            total_recon_loss += recon_loss.detach().item()

            #######################
            # Regularization phase
            #######################
            zs_real, labels_real = self.prior.sample(batch_size, with_classes=True)
            zs_real = zs_real.to(self.device)
            labels_real = labels_real.to(self.device)
            labels_real = self._random_label_flip(labels_real)
            ys_real = F.one_hot(labels_real, self.n_classes + 1)
            D_real_inputs = torch.cat((zs_real, ys_real), 1)


            # Discriminator
            self.Q.eval()
            self.D_gauss.train()
            self.disc_optim.zero_grad()
            with torch.set_grad_enabled(True):
                zs = self.Q(inputs)
                ys = F.one_hot(labels, self.n_classes + 1)
                D_inputs = torch.cat((zs, ys), 1)

                D_real_gauss = self.D_gauss(D_real_inputs)
                D_fake_gauss = self.D_gauss(D_inputs)

                D_loss_gauss = -6 * torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

                D_loss = D_loss_gauss

                D_loss.backward()
                self.disc_optim.step()
            total_D_loss += D_loss.detach().item()

            # Generator
            self.Q.train()
            self.D_gauss.eval()
            self.gen_optim.zero_grad()
            with torch.set_grad_enabled(True):
                zs = self.Q(inputs)
                ys = F.one_hot(labels, self.n_classes + 1)
                D_inputs = torch.cat((zs, ys), 1)

                D_fake_gauss = self.D_gauss(D_inputs)

                G_loss = -6 * torch.mean(torch.log(D_fake_gauss + EPS))
                G_loss.backward()
                self.gen_optim.step()
            total_G_loss += G_loss.detach().item()
        
        l = len(dataloader)
        return total_recon_loss / l, total_D_loss / l, total_G_loss / l
