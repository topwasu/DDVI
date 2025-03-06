import logging
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from diffusion.gaussian import GaussianDiffusionNoise
from models.simple import ConditionalModel, DoubleConditionalModel
from utils import visualize_latent


log = logging.getLogger(__name__)


class StandardDiffusion():
    def __init__(self, config):
        self.prior = config.prior
        self.latent_size = config.latent_size
        self.num_epochs = config.num_epochs
        self.timesteps = config.timesteps
        self.lr = config.lr
        self.batch_size = config.batch_size

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.init_models_and_optims()
        
        self.save_folder = config.save_folder

    def init_models_and_optims(self):
        self.Q = GaussianDiffusionNoise(ConditionalModel(self.latent_size, self.timesteps), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=torch.device('cuda'))

        self.Q.to(self.device)


        # Set optimizators
        self.optim = optim.Adam(self.Q.parameters(), lr=self.lr)


    def train(self, *args):
        log.info("Start training")
        start = time.time()
        for epoch in range(self.num_epochs):
            if epoch % 10 == 0:
                log.info(f'Epoch: {epoch}')
            self.ep = epoch
            res = self._train_epoch()

            if epoch % 10 == 0:
                log.info(f'Current time {time.time() - start}')
                log.info(f'Losses {res}')
                latents = self.eval()
                if not os.path.exists(self.save_folder):
                    os.makedirs(self.save_folder)
                visualize_latent(latents[:, :2], os.path.join(self.save_folder, f'latent_ep{epoch}.png'))

    def _train_epoch(self):
        '''
        Train procedure for one epoch.
        '''
        n_its = 1000

        all_res = 0
        ct = 0
        for _ in range(n_its):
            prior_samples = self.prior.sample(self.batch_size)

            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                t = torch.randint(
                    0, self.timesteps, (self.batch_size,), device=self.device
                ).long()

                diff_loss = self.Q.loss_at_step_t(prior_samples, t, loss_type="huber")

                loss = diff_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([diff_loss.detach().item()])
            all_res += res
        
        return all_res / n_its

    def eval(self, *args):
        self.Q.eval()
        reps = []
        for _ in range(100):
            with torch.set_grad_enabled(False):
                zs = self.Q.sample(self.batch_size)
                rep = zs[-1]
                reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])


class ClassDiffusion(StandardDiffusion):
    def init_models_and_optims(self):
        self.Q = GaussianDiffusionNoise(DoubleConditionalModel(self.latent_size, self.timesteps), 
                                        self.timesteps,
                                        self.latent_size,
                                        device=torch.device('cuda'))

        self.Q.to(self.device)


        # Set optimizators
        self.optim = optim.Adam(self.Q.parameters(), lr=self.lr)


    def train(self, *args):
        log.info("Start training")
        start = time.time()
        for epoch in range(self.num_epochs):
            if epoch % 10 == 0:
                log.info(f'Epoch: {epoch}')
            self.ep = epoch
            res = self._train_epoch()

            if epoch % 10 == 0:
                log.info(f'Current time {time.time() - start}')
                log.info(f'Losses {res}')
                latents = self.eval()
                classes = np.tile(np.arange(10), self.batch_size * 10)
                if not os.path.exists(self.save_folder):
                    os.makedirs(self.save_folder)
                visualize_latent(latents, os.path.join(self.save_folder, f'latent_ep{epoch}.png'), targets=classes)

    def _train_epoch(self):
        '''
        Train procedure for one epoch.
        '''

        all_res = 0
        ct = 0
        for _ in range(1000):
            prior_samples, classes = self.prior.sample(self.batch_size, with_classes=True)

            # -------------- Reconstruction --------------------
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                t = torch.randint(
                    0, self.timesteps, (self.batch_size,), device=self.device
                ).long()

                diff_loss = self.Q.loss_at_step_t(prior_samples, t, aux=classes, loss_type="huber")

                loss = diff_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([diff_loss.detach().item()])
            all_res += res
        
        l = 500
        return all_res / l

    def eval(self, *args):
        self.Q.eval()
        reps = []
        classes = torch.Tensor(np.tile(np.arange(10), self.batch_size * 10)).long().to(self.device)
        for i in range(100):
            cur_classes = classes[i*self.batch_size: (i+1)*self.batch_size]
            with torch.set_grad_enabled(False):
                zs = self.Q.sample(self.batch_size, aux=cur_classes)
                rep = zs[-1]
                reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)
    
    def save(self, path):
        torch.save({
            'Q': self.Q.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])