"""Implements the core diffusion algorithms."""
import logging
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx


log = logging.getLogger(__name__)


class GaussianDiffusionMean(nn.Module):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, model, timesteps, img_shape, schedule='linear', device='cpu'
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.img_shape = img_shape
        self.schedule = get_schedule(schedule)
        self.device=device

        # initialize the alpha and beta paramters
        self.betas = self.schedule(timesteps) # TODO: careful, this starts at 0
        self.alphas = 1 - self.betas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.bar_alphas = torch.cumprod(self.alphas, axis=0)
        self.sqrt_bar_alphas = torch.sqrt(self.bar_alphas)
        self.sqrt_one_minus_bar_alphas = torch.sqrt(1. - self.bar_alphas)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        bar_alphas_prev = F.pad(self.bar_alphas[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1. - bar_alphas_prev) / (1. - self.bar_alphas)
        )

        # TODO: make sure these are correct
        self.mean_weight_t = torch.sqrt(self.alphas) * (1. - bar_alphas_prev) / (1. - self.bar_alphas)
        self.mean_weight_0 = self.betas * torch.sqrt(bar_alphas_prev) / (1. - self.bar_alphas)
        self.mean_weight_0, self.mean_weight_t = self.mean_weight_0.to(self.device), self.mean_weight_t.to(self.device)

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.
        Takes a sample from q(xt|x0).
        """
        if noise is None: noise = torch.randn_like(x0)

        sqrt_bar_alphas_t = get_by_idx(
            self.sqrt_bar_alphas, t, x0.shape
        )
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, x0.shape
        )

        return (
            sqrt_bar_alphas_t * x0 
            + sqrt_one_minus_bar_alphas_t * noise
        )
    
    def p_sample(self, xt, t, aux=None, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Takes a sample from p(x_{t-1}|x_t).
        """
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        inputs = [xt, t] if aux is None else [xt, aux, t]
        xt_prev_mean = self.model(*inputs)

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt, device=self.device)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + torch.sqrt(post_var_t) * noise

        # return x_{t-1}
        return xt_prev, xt_prev_mean


    def sample(self, batch_size, x=None, aux=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        if isinstance(self.img_shape, int):
            shape = (batch_size, self.img_shape)
        else:
            shape = (batch_size, *self.img_shape)
        if x is None: 
            x = torch.randn(shape, device=self.device)
        xs = torch.zeros(self.timesteps + 1, *shape, device=self.device)
        xs[self.timesteps] = x
        mean_xs = torch.zeros(self.timesteps + 1, *shape, device=self.device)

        for t in reversed(range(0, self.timesteps)):
            T = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            # if t == 0:
            #     x, mean_x, log_var_x = self.p_sample(x, T, aux=aux, deterministic=deterministic)
            # else:
            x, mean_x = self.p_sample(x, T, aux=aux, deterministic=deterministic)
            xs[t] = x
            mean_xs[t] = mean_x
        return xs, mean_xs
    
    def get_posterior_means_and_variances(self, samples):
        # TODO: double check this
        # the weights are 1D vector of length `timestep` and samples' shape is (timestep + 1, batch_size, latent_size)
        means = self.mean_weight_0.view(-1, 1, 1) * samples[0].expand((self.timesteps, -1, -1)) + self.mean_weight_t.view(-1, 1, 1) * samples[1:] # TODO: check this
        # means = F.pad(means[-1:], (0, 1), value=1.0)
        means = torch.cat((means, (self.sqrt_bar_alphas[-1] * samples[0])[None, :]))
        return means, torch.cat((self.posterior_variance,  torch.tensor([1. - self.bar_alphas[-1]]))).to(self.device)
    
    def get_posterior_mean(self, sample_0, sample_t, t):
        raise NotImplementedError

    def p_loss_at_step_t(self, noise, predicted_noise, loss_type="l1"):
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    def loss_at_step_t(self, x0, t, aux=None, loss_type="l2", noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0=x0, t=t, noise=noise)

        # understand this - should be okay
        _, predicted_mean = self.p_sample(x_t, t, aux, True)
        pos_mean = self.get_posterior_mean(x0, x_t, t)
        loss = self.p_loss_at_step_t(pos_mean.detach(), predicted_mean, loss_type)
        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, eval=False):
        self.load_state_dict(torch.load(path))
        if eval:
            self.model.eval()


class GaussianDiffusionNoise(nn.Module):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, model, timesteps, img_shape, schedule='linear', device='cpu'
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.img_shape = img_shape
        self.schedule = get_schedule(schedule)
        self.device=device

        # initialize the alpha and beta paramters
        self.betas = self.schedule(timesteps)
        self.alphas = 1 - self.betas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        bar_alphas = torch.cumprod(self.alphas, axis=0)
        self.sqrt_bar_alphas = torch.sqrt(bar_alphas)
        self.sqrt_one_minus_bar_alphas = torch.sqrt(1. - bar_alphas)
        self.sqrt_one_minus_bar_alphas_prev = F.pad(self.sqrt_one_minus_bar_alphas[:-1], (1, 0), value=0.0)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        bar_alphas_prev = F.pad(bar_alphas[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1. - bar_alphas_prev) / (1. - bar_alphas)
        )

        self.mean_weight_t = torch.sqrt(self.alphas) * (1. - bar_alphas_prev) / (1. - bar_alphas)
        self.mean_weight_0 = self.betas * torch.sqrt(bar_alphas_prev) / (1. - bar_alphas)

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.
        Takes a sample from q(xt|x0).
        """
        if noise is None: noise = torch.randn_like(x0)

        sqrt_bar_alphas_t = get_by_idx(
            self.sqrt_bar_alphas, t, x0.shape
        )
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, x0.shape
        )

        return (
            sqrt_bar_alphas_t * x0 
            + sqrt_one_minus_bar_alphas_t * noise
        )


    def p_sample(self, xt, t, aux=None, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.
        Takes a sample from p(x_{t-1}|x_t).
        """
        betas_t = get_by_idx(self.betas, t, xt.shape)
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, xt.shape
        )
        sqrt_one_minus_bar_alphas_prev_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas_prev, t, xt.shape
        )
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, t, xt.shape)
        
        # From https://arxiv.org/pdf/2105.05233.pdf Algorithm 2
        inputs = [xt, t] if aux is None else [xt, aux, t]
        predicted_noise = self.model(*inputs)
        scaled_predicted_x0 = sqrt_recip_alphas_t * (xt - sqrt_one_minus_bar_alphas_t * predicted_noise)
        direction = sqrt_one_minus_bar_alphas_prev_t * predicted_noise

        return scaled_predicted_x0 + direction


    def sample(self, batch_size, aux=None, x=None, deterministic=False, get_xt=False):
        """Samples from the diffusion process, producing images from noise
        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        if isinstance(self.img_shape, int):
            shape = (batch_size, self.img_shape)
        else:
            shape = (batch_size, *self.img_shape)
        
        if x is None: 
            x = torch.randn(shape, device=self.device)
        xs = []

        if get_xt:
            xs.append(x)

        for t in reversed(range(0, self.timesteps)):
            T = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, T, aux=aux, deterministic=(t==0 or deterministic))
            xs.append(x)
        return xs

    def p_loss_at_step_t(self, noise, predicted_noise, loss_type="l1", nored=False):
        reduction = 'none' if nored else 'mean'
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise, reduction=reduction)
        else:
            raise NotImplementedError()
        return loss

    def loss_at_step_t(self, x0, t, aux=None, loss_type="l1", noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)

        if aux is None:
            predicted_noise = self.model(x_noisy, t)
        else:
            predicted_noise = self.model(x_noisy, aux, t)

        loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)

        return loss
    
    def weird_loss_at_step_t(self, x0, xt, t, aux, loss_type="l1", nored=False):
        w0 = get_by_idx(self.mean_weight_0, t, x0.shape)
        wt = get_by_idx(self.mean_weight_t, t, xt.shape)
        mean = w0 * x0 + wt * xt
        predicted_mean = self.p_sample(xt, t, aux)

        loss = self.p_loss_at_step_t(mean, predicted_mean, loss_type, nored=nored)

        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, eval=False):
        self.load_state_dict(torch.load(path))
        if eval:
            self.model.eval()

    def kl_divergence_at_t(self, x0, xt, t, mu1, nored=False):
        w0 = get_by_idx(self.mean_weight_0, t, x0.shape)
        wt = get_by_idx(self.mean_weight_t, t, xt.shape)
        mu2 = w0 * x0 + wt * xt

        post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)

        return kl_divergence_gaussian(mu1, torch.sqrt(post_var_t), mu2, torch.sqrt(post_var_t), nored)


def kl_divergence_gaussian(mu1, std1, mu2, std2, nored=False):
    """
    Calculate the KL divergence between two Gaussian distributions.

    :param mu1: Mean of the first Gaussian distribution.
    :param std1: Standard deviation of the first Gaussian distribution.
    :param mu2: Mean of the second Gaussian distribution.
    :param std2: Standard deviation of the second Gaussian distribution.
    :return: KL divergence value.
    """
    var1 = std1.pow(2)
    var2 = std2.pow(2)
    kl_div = torch.log(std2/std1) + (var1 + (mu1 - mu2).pow(2)) / (2 * var2) - 0.5
    if nored:
        return kl_div.sum(-1)
    else:
        return kl_div.mean()


class TwoDiffusion():
    def __init__(self, g, g2, timesteps, img_shape, schedule='linear', device='cpu', g2_weight=1):
        self.g = g
        self.g2 = g2
        self.g2_weight = g2_weight

        self.timesteps = timesteps
        self.img_shape = img_shape
        self.schedule = get_schedule(schedule)
        self.device=device

        # initialize the alpha and beta paramters
        self.betas = self.schedule(timesteps)
        self.alphas = 1 - self.betas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        bar_alphas = torch.cumprod(self.alphas, axis=0)
        self.sqrt_bar_alphas = torch.sqrt(bar_alphas)
        self.sqrt_one_minus_bar_alphas = torch.sqrt(1. - bar_alphas)
        self.sqrt_one_minus_bar_alphas_prev = F.pad(self.sqrt_one_minus_bar_alphas[:-1], (1, 0), value=0.0)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        bar_alphas_prev = F.pad(bar_alphas[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1. - bar_alphas_prev) / (1. - bar_alphas)
        )

        self.mean_weight_t = torch.sqrt(self.alphas) * (1. - bar_alphas_prev) / (1. - bar_alphas)
        self.mean_weight_0 = self.betas * torch.sqrt(bar_alphas_prev) / (1. - bar_alphas)

    def p_sample(self, xt, t, aux=None, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.
        Takes a sample from p(x_{t-1}|x_t).
        """
        betas_t = get_by_idx(self.betas, t, xt.shape)
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, xt.shape
        )
        sqrt_one_minus_bar_alphas_prev_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas_prev, t, xt.shape
        )
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, t, xt.shape)
        
        # From https://arxiv.org/pdf/2105.05233.pdf Algorithm 2
        if self.g2_weight == 1:
            predicted_noise = self.g2_weight * self.g2.model(*[xt, aux, t])
        else:
            predicted_noise = (1 - self.g2_weight) * self.g.model(*[xt, t]) + self.g2_weight * self.g2.model(*[xt, aux, t])
        scaled_predicted_x0 = sqrt_recip_alphas_t * (xt - sqrt_one_minus_bar_alphas_t * predicted_noise)
        direction = sqrt_one_minus_bar_alphas_prev_t * predicted_noise

        return scaled_predicted_x0 + direction
    
    def sample(self, batch_size, aux=None, x=None, deterministic=False, get_xt=False):
        """Samples from the diffusion process, producing images from noise
        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        if isinstance(self.img_shape, int):
            shape = (batch_size, self.img_shape)
        else:
            shape = (batch_size, *self.img_shape)
        
        if x is None: 
            x = torch.randn(shape, device=self.device)
        xs = []

        if get_xt:
            xs.append(x)

        for t in reversed(range(0, self.timesteps)):
            T = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, T, aux=aux, deterministic=(t==0 or deterministic))
            xs.append(x)
        return xs
    
    def kl_divergence_at_t(self, x0, xt, t, mu1, nored=False):
        w0 = get_by_idx(self.mean_weight_0, t, x0.shape)
        wt = get_by_idx(self.mean_weight_t, t, xt.shape)
        mu2 = w0 * x0 + wt * xt

        post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)

        return kl_divergence_gaussian(mu1, torch.sqrt(post_var_t), mu2, torch.sqrt(post_var_t), nored)
    
