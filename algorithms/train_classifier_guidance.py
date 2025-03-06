import logging
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from diffusion.guidance import GuidedDiffWrapper
from models.simple import TimeClassifier
from utils import visualize_latent
from priors import get_prior

log = logging.getLogger(__name__)


class ClassifierGuidance():
    def __init__(self, Q, config):
        self.Q = Q

        self.prior = config.prior
        self.latent_size = config.latent_size
        self.num_epochs = config.num_epochs
        self.timesteps = config.timesteps
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.n_classes = config.n_classes

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.init_models_and_optims()

        self.save_folder = config.save_folder

    def init_models_and_optims(self):
        self.classifier = TimeClassifier(self.latent_size, self.n_classes, self.timesteps)
        self.classifier.to(self.device)

        self.Q.to(self.device)

        self.guided_Q = GuidedDiffWrapper(self.Q, self.classifier, self.timesteps, self.latent_size)

        # Set optimizators
        self.optim = optim.Adam(self.classifier.parameters(), lr=self.lr)


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
                latents, targets = self.eval()
                if not os.path.exists(self.save_folder):
                    os.makedirs(self.save_folder)
                visualize_latent(latents[:, :2], os.path.join(self.save_folder, f'latent_ep{epoch}.png'), targets=targets)

    def _train_epoch(self):
        '''
        Train procedure for one epoch.
        '''
        n_its = 1000

        all_res = 0
        for _ in range(n_its):
            prior_samples, targets = self.prior.sample(self.batch_size, with_classes=True)

            # -------------- Reconstruction --------------------
            self.Q.train()
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                t = torch.randint(
                    0, self.timesteps, (self.batch_size,), device=self.device
                ).long()

                diff_loss = self.guided_Q.classifier_loss_at_t(prior_samples, t, targets)

                loss = diff_loss
                loss.backward()
                self.optim.step()
            res = np.asarray([diff_loss.detach().item()])
            all_res += res
        
        return all_res / n_its

    def eval(self, *args):
        self.Q.eval()
        reps = []
        
        targets = torch.arange(self.n_classes).repeat(1000).to(self.device)

        for i in range(0, 10000, 10 * self.n_classes):
            with torch.set_grad_enabled(True):
                zs = self.guided_Q.sample(targets[i:i + 10*self.n_classes])
                rep = zs[-1]
                reps.append(rep.detach().cpu().numpy())
        return np.concatenate(reps, axis=0), targets.detach().cpu().numpy()
    
    def save(self, path):
        torch.save({
            'Q': self.guided_Q.Q.state_dict(),
            'classifier': self.guided_Q.classifier.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.guided_Q.Q.load_state_dict(checkpoint['Q'])
        self.guided_Q.classifier.load_state_dict(checkpoint['classifier'])