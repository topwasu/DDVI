import torch
import torch.nn.functional as F
import logging


log = logging.getLogger(__name__)


def sample(mu, log_var, ret_log_prob=False):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    if ret_log_prob:
        return  mu + eps * std, -torch.sum(log_var + (eps ** 2) / 2 + torch.log(torch.tensor(2 * 3.14)) / 2, -1)
    return  mu + eps * std

def sample_k(mu, log_var, k, ret_log_prob=False):
    std = torch.exp(0.5 * log_var)

    mu = mu.unsqueeze(1).repeat(1, k, 1)
    log_var = log_var.unsqueeze(1).repeat(1, k, 1)
    std = std.unsqueeze(1).repeat(1, k, 1)
    eps = torch.randn_like(std)
    if ret_log_prob:
        return  mu + eps * std, -torch.sum(log_var + (eps ** 2) / 2 + torch.log(torch.tensor(2 * 3.14)) / 2, -1)
    return  mu + eps * std


def sample_from_dirichlet(alphas, gauss_softmax=False, n_clusters=20):
    if gauss_softmax:
        log_alphas = torch.log(alphas)
        mean_log_alphas = torch.mean(log_alphas, -1, keepdim=True)
        mu = log_alphas - mean_log_alphas

        k1 = 1 - (2 / n_clusters)
        k2 = 1 / (n_clusters ** 2)
        sigma = k1 * 1/alphas + k2 * torch.sum(1./alphas, -1, keepdim=True)

        # log.info(f'mu {mu} sigma {sigma} alphas {alphas} log_alphas {log_alphas}')

        eps = torch.randn_like(sigma)
        ys = F.softmax(mu + sigma * eps, -1)
        # log.info(f'ys {ys}')
    else:
        u = torch.rand_like(alphas)
        xs = torch.pow(u * alphas * torch.exp(torch.lgamma(alphas)), 1. / alphas) + 1e-8
        ys = xs / torch.sum(xs, -1)[:, None]
        ys = torch.clamp(ys, 1e-6)
    return  ys


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    h = 0.3
    numerator = ((a_core - b_core)/h).pow(2).mean(2)/depth / h
    return torch.exp(-numerator)


def mmd_func(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()


class DiffusionConfig:
    batch_size = 128
    lr = 1e-3
    timesteps = 10
    num_epochs = 200