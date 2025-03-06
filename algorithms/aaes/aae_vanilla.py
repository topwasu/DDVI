"""Ref: https://github.com/fducau/AAE_pytorch"""
import logging
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_model import BaseModel
from models.simple import MultiWayLinear
from priors import get_prior
from .utils import GaussianNoise
from ..utils import AddNoise, get_q_and_p


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


class AAE_vanilla(BaseModel):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset.name
        self.prior = get_prior(config)
        self.latent_size = config.model.latent_size
        self.num_epochs = config.model.num_epochs if config.dataset.name.startswith('mnist') else 100
        self.hidden_size = config.model.hidden_size
        self.bn = config.model.bn
        self.lr = config.model.lr
        self.batch_size = config.model.batch_size

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Set loss
        self.criterion = torch.nn.MSELoss()

        self.save_folder = config.save_folder

        self.init_models_and_optims()

    def init_models_and_optims(self):
        self.Q, self.P = get_q_and_p(self.config)

        self.D_gauss = D_net(self.latent_size)
        
        self.Q.to(self.device)
        self.P.to(self.device)
        self.D_gauss.to(self.device)

        # Set optimizators
        self.ae_optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), 
                                       lr=self.lr)
        self.ae_scheduler = optim.lr_scheduler.MultiStepLR(self.ae_optim, 
                                                           milestones=[150], 
                                                           gamma=0.1)

        self.disc_optim = optim.Adam(self.D_gauss.parameters(), lr=self.lr)
        self.disc_scheduler = optim.lr_scheduler.MultiStepLR(self.disc_optim, 
                                                             milestones=[150], 
                                                             gamma=0.1)
        
        self.gen_optim = optim.Adam(self.Q.parameters(), lr=self.lr)
        self.gen_scheduler = optim.lr_scheduler.MultiStepLR(self.gen_optim, 
                                                            milestones=[150], 
                                                            gamma=0.1)
        
    def _call_schedulers(self):
        self.ae_scheduler.step()
        self.disc_scheduler.step()
        self.gen_scheduler.step()
        self.D_gauss.noise.sigma = max(0, 0.1 * (((self.num_epochs - (self.ep + 1)) // (self.num_epochs // 4)) - 1))

    def _train_epoch(self, dataloader):
        '''
        Train procedure for one epoch.
        '''
        EPS = 1e-15

        total_recon_loss, total_D_loss, total_G_loss = 0, 0, 0
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
                zs = self.Q(inputs)
                inputs_reconstructed = self.P(zs)

                recon_loss = 0.5 * self.criterion(inputs_reconstructed, inputs)
                recon_loss.backward()
                self.ae_optim.step()
            total_recon_loss += recon_loss.detach().item()

            #######################
            # Regularization phase
            #######################
            zs_real = self.prior.sample(batch_size)
            zs_real = zs_real.to(self.device)

            # Discriminator
            self.Q.eval()
            self.D_gauss.train()
            self.disc_optim.zero_grad()
            with torch.set_grad_enabled(True):
                zs = self.Q(inputs)

                D_real_gauss = self.D_gauss(zs_real)
                D_fake_gauss = self.D_gauss(zs)

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

                D_fake_gauss = self.D_gauss(zs)

                G_loss = -6 * torch.mean(torch.log(D_fake_gauss + EPS))
                G_loss.backward()
                self.gen_optim.step()
            total_G_loss += G_loss.detach().item()
        
        l = len(dataloader)
        return total_recon_loss / l, total_D_loss / l, total_G_loss / l

    def eval(self, dataloader):
        self.Q.eval()
        reps = []
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                zs = self.Q(inputs)
                rep = zs
            reps.append(rep.cpu().detach().numpy())
        return np.concatenate(reps, axis=0)

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
            'P': self.P.state_dict(),
            'D_gauss': self.D_gauss.state_dict()
            }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])
        self.P.load_state_dict(checkpoint['P'])
        self.D_gauss.load_state_dict(checkpoint['D_gauss'])

    def generate(self, n):
        all_outputs = []
        for i in range(0, n, self.batch_size):
            samples = self.prior.sample(min(self.batch_size, n-i))
            outputs = self.P(samples)
            all_outputs.append(outputs.cpu().detach().numpy())
        return np.concatenate(all_outputs, axis=0)
