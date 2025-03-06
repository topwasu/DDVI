import numpy as np
import torch
import logging
from .schedule import get_schedule, get_by_idx


log = logging.getLogger(__name__)

class GuidedDiffWrapper:
    def __init__(self, Q, classifier, timesteps, latent_size):
        self.Q = Q
        self.classifier = classifier

        self.timesteps = timesteps
        self.latent_size = latent_size

        self.criterion = torch.nn.CrossEntropyLoss()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.classifier_scale = np.ones(self.timesteps) * 2 # TODO: do something with this
        self.loss_scale = 1 # TODO: do something with this
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def classifier_loss_at_t(self, x0, t, inputs):
        noise = torch.randn_like(x0)
        x_noisy = self.Q.q_sample(x0=x0, t=t, noise=noise)

        inputs_reconstructed = self.classifier(x_noisy, t)

        loss = self.criterion(inputs_reconstructed, inputs) * self.loss_scale

        return loss
    
    def classifier_gradient_at_t(self, x, t, inputs):
        x_in = x.detach().requires_grad_(True) # TODO: detach...?
        inputs_reconstructed = self.classifier(x_in, t) # TODO: maybe t + 1

        p = self.log_softmax(inputs_reconstructed)
        selected = p[range(len(inputs_reconstructed)), inputs.view(-1)]

        return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale[t[0]]

    
    def sample(self, inputs, x=None):
        batch_size = inputs.size()[0]
        if x is None: 
            x = torch.randn((batch_size, self.latent_size), device=self.device)
        xs = []

        for t in reversed(range(0, self.timesteps)):
            T = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            grad = self.classifier_gradient_at_t(x, T, inputs)
            x = self.Q.p_sample(x, T)

            # one_minus_bar_alphas = 1. - self.Q.bar_alphas
            sqrt_recip_alphas = torch.sqrt(1.0 / self.Q.alphas)
            # one_minus_bar_alphas_t = get_by_idx(one_minus_bar_alphas, T, x.shape)
            sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, T, x.shape)
            sqrt_one_minus_bar_alphas_t = get_by_idx(
                self.Q.sqrt_one_minus_bar_alphas, T, x.shape
            )
            sqrt_one_minus_bar_alphas_prev_t = get_by_idx(
                self.Q.sqrt_one_minus_bar_alphas_prev, T, x.shape
            )

            x = x + sqrt_one_minus_bar_alphas_t * sqrt_one_minus_bar_alphas_t  * sqrt_recip_alphas_t * grad \
                - sqrt_one_minus_bar_alphas_t * sqrt_one_minus_bar_alphas_prev_t * grad
            xs.append(x)
        return xs