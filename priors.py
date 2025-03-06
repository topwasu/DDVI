import logging
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch_kde import GaussianKernel, KernelDensityEstimator


log = logging.getLogger(__name__)


class BasePrior:
    def __init__(self, latent_size):
        self.latent_size = latent_size
        self.kdes = []
        self.partition_kdes = None

    def sample(self, batch_size, g_classes=None, with_classes=False):
        raise NotImplementedError
    
    def evaluate(self, zs, nored=False):
        potential_sigmas = np.asarray([0.005, 0.008, 0.01, 0.03, 0.05])
        if len(self.kdes) == 0:
            prior_samples = self.sample(10000)
            for sigma in potential_sigmas:
                kde = KernelDensityEstimator(prior_samples, GaussianKernel(sigma))
                self.kdes.append(kde)

        all_log_scores = []
        for kde in self.kdes:
            log_scores = torch.clamp(kde.forward(zs), -1000, 1000)
            all_log_scores.append(log_scores)
        all_log_scores = torch.stack(all_log_scores)
        if nored:
            return torch.logsumexp(all_log_scores, 0) - torch.log(torch.tensor(len(potential_sigmas)))
        return torch.mean(torch.logsumexp(all_log_scores, 0) - torch.log(torch.tensor(len(potential_sigmas))))
    
    def evaluate_partition(self, zs, partition):
        potential_sigmas = np.asarray([0.05])
        if self.partition_kdes is None:
            self.partition_kdes = []
            for i in range(self.n_classes):
                kdes = []
                prior_samples = self.sample(2000, g_classes=np.asarray([i] * 2000))
                for sigma in potential_sigmas:
                    kde = KernelDensityEstimator(prior_samples, GaussianKernel(sigma))
                    kdes.append(kde)
                self.partition_kdes.append(kdes)

        all_log_scores = []
        for kde in self.partition_kdes[partition]:
            log_scores = torch.clamp(kde.forward(zs), -300, 300)
            all_log_scores.append(log_scores)
        all_log_scores = torch.stack(all_log_scores)
        return torch.logsumexp(all_log_scores, 0) - torch.log(torch.tensor(len(potential_sigmas)))


class GaussPrior(BasePrior):
    def __init__(self, latent_size, std=1.):
        super().__init__(latent_size)
        self.mean = torch.tensor([0.])
        self.std = torch.tensor([std])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
        self.gaussian = Normal(self.mean, self.std)

    def sample(self, batch_size, g_classes=None, with_classes=False):
        if g_classes is not None or with_classes:
            raise NotImplementedError
        return torch.randn(batch_size, self.latent_size, device=self.device) * self.std
        
    def evaluate(self, z):
        batch_shape = z.size()

        return torch.mean(-torch.log(torch.tensor(2 * 3.14)) - torch.sum(torch.log(torch.tensor(self.std ** 2))) / 2- torch.sum((z ** 2) / (self.std ** 2), -1) / 2)


class GaussMixPrior(BasePrior):
    def __init__(self, latent_size, n_classes=10):
        # https://github.com/nicklhy/AdversarialAutoEncoder/blob/master/data_factory.py#L40
        super().__init__(latent_size)
        if latent_size != 2:
            raise NotImplementedError
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_classes = n_classes
        self.x_var = 0.5 ** 2
        self.y_var = 0.1 ** 2

        self.gaussians = []
        shift = 1.4
        for i in range(10):
            mean = np.zeros(2)
            covariance = np.diag([self.x_var, self.y_var])
            r = 2.0 * np.pi / 10.0 * i
            rotation_mat = np.asarray([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
            covariance = rotation_mat @ covariance @ rotation_mat.T
            mean += shift * np.asarray([np.cos(r), np.sin(r)])
            mean = torch.tensor(mean, device=self.device)
            covariance = torch.tensor(covariance, device=self.device)
            self.gaussians.append(MultivariateNormal(mean, covariance))


    def sample(self, batch_size, g_classes=None, with_classes=False):
        if g_classes is None:
            g_classes = np.random.randint(self.n_classes, size=batch_size)
        # g_classes = np.ones(batch_size)
        xs = np.random.normal(0, np.sqrt(self.x_var), batch_size)
        ys = np.random.normal(0, np.sqrt(self.y_var), batch_size)
        shift = 1.4
        r = 2.0 * np.pi / self.n_classes * g_classes
        new_xs = xs * np.cos(r) - ys * np.sin(r)
        new_ys = xs * np.sin(r) + ys * np.cos(r)
        new_xs += shift * np.cos(r)
        new_ys += shift * np.sin(r)
        zs = np.stack((new_xs, new_ys), axis=-1)

        if with_classes:
            return torch.Tensor(zs).to(self.device), torch.Tensor(g_classes).long().to(self.device)
        else:
            return torch.Tensor(zs).to(self.device)
        
    def evaluate(self, z):
        batch_size = z.size()[:1]

        p = []
        for i in range(10):
            p.append(self.gaussians[i].expand(batch_size).log_prob(z))
        p = torch.stack(p, -1)
        p = torch.logsumexp(p, -1) - torch.log(torch.tensor([10]).to(self.device))
        return p
    

class SwissRollPrior(BasePrior):
    def __init__(self, latent_size, n_classes=10, noise_level=0, mult=None):
        super().__init__(latent_size)
        self.latent_size = latent_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_classes = n_classes

        self.noise_level = noise_level
        self.mult = mult if mult is not None else 1

    def sample(self, batch_size, g_classes=None, with_classes=False):
        if self.latent_size > 2:
            if g_classes is not None or with_classes:
                raise Exception("sample_highd does not support g_classes or with_classes arguemnts")
            return self.sample_highd(batch_size)
        
        if g_classes is None:
            g_classes = np.random.randint(self.n_classes, size=batch_size)

        uni = np.random.uniform(0.0, 1.0, size=batch_size) / self.n_classes + g_classes / self.n_classes
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 4.5 * np.sqrt(uni)
        xs = r * np.cos(rad)
        ys = r * np.sin(rad)

        zs = (np.stack((xs, ys), axis=-1) + np.random.randn(batch_size, 2) * self.noise_level) * self.mult
        
        if with_classes:
            return torch.Tensor(zs).to(self.device), torch.Tensor(g_classes).long().to(self.device)
        else:
            return torch.Tensor(zs).to(self.device)
    
    def sample_highd(self, batch_size, g_classes=None, with_classes=False):
        if g_classes is None:
            g_classes = np.random.randint(self.n_classes, size=batch_size)

        uni = np.random.uniform(0.0, 1.0, size=batch_size) / self.n_classes + g_classes / self.n_classes
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 4.5 * np.sqrt(uni)
        xs = r * np.cos(rad)
        ys = r * np.sin(rad)

        zs = np.zeros((batch_size, self.latent_size))
        zs[:, 0] = xs
        zs[:, 1] = ys
        zs = (zs + np.random.randn(batch_size, self.latent_size) * self.noise_level) * self.mult
        
        if with_classes:
            return torch.Tensor(zs).to(self.device), torch.Tensor(g_classes).long().to(self.device)
        else:
            return torch.Tensor(zs).to(self.device)

class GridGaussPrior:
    def __init__(self, latent_size, n_classes, space=8):
        if latent_size < 2:
            raise NotImplementedError('Latent size has to be at least 2')
        super().__init__(latent_size)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_classes = n_classes

        cols = int(np.sqrt(self.n_classes))
        rows = -(-self.n_classes // cols)
        self.scale = 4 / ((max(cols, rows) - 1) * space)

        self.means = torch.zeros(self.n_classes, self.latent_size)
        for i in range(self.n_classes):
            x = ((i // cols) * space) - ((rows - 1) * space) // 2
            y = ((i % cols) * space) - ((cols - 1) * space) // 2
            scaled_x = self.scale * x
            scaled_y = self.scale * y
            self.means[i] = torch.Tensor(np.concatenate(([scaled_x, scaled_y], np.zeros(self.latent_size - 2))))

    def sample(self, batch_size, g_classes=None, with_classes=False):
        if g_classes is None:
            g_classes = np.random.randint(self.n_classes, size=batch_size)

        unshifted_samples = torch.randn(batch_size, self.latent_size) * self.scale
        shifted_samples = self.means[g_classes] + unshifted_samples

        if with_classes:
            return torch.Tensor(shifted_samples).to(self.device), torch.Tensor(g_classes).long().to(self.device)
        else:
            return torch.Tensor(shifted_samples).to(self.device)
        
    def evaluate_class(self, z):
        log_ps = []
        for i in range(self.n_classes):
            log_p = -torch.sum(torch.square(z - self.means[i]), -1)
            log_ps.append(log_p.view(-1, 1))
        log_ps = torch.cat(log_ps, -1)
        log_ps = torch.exp(log_ps - torch.logsumexp(log_ps, 1).view(-1, 1))
        return log_ps
    

class PinWheelPrior(BasePrior):
    def __init__(self, latent_size, n_classes=10, mult=None):
        super().__init__(latent_size)
        if latent_size != 2:
            raise NotImplementedError
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_classes = n_classes
        # self.radial_std =  0.3
        self.radial_std =  0.15
        # self.tangential_std = 0.05
        self.tangential_std = 0.005
        self.rate = 0.25

        self.mult = mult if mult is not None else 1

    def sample(self, batch_size, g_classes=None, with_classes=False):
        if g_classes is None:
            g_classes = np.random.randint(self.n_classes, size=batch_size)
        
        rads = np.linspace(0, 2*np.pi, self.n_classes, endpoint=False)

        features = np.random.randn(batch_size, 2) * np.array([self.radial_std, self.tangential_std])
        features[:,0] += 1.

        angles = rads[g_classes] + self.rate * np.exp(features[:,0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        zs = np.einsum('ti,tij->tj', features, rotations) * self.mult
    
        if with_classes:
            return torch.Tensor(zs).to(self.device), torch.Tensor(g_classes).long().to(self.device)
        else:
            return torch.Tensor(zs).to(self.device)
        

class SquarePrior(BasePrior):
    def __init__(self, latent_size, n_classes=10, noise_level=0, mult=None):
        super().__init__(latent_size)
        self.latent_size = latent_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_classes = n_classes
        self.noise_level = noise_level

        self.mult = mult if mult is not None else 1

    def sample(self, batch_size, g_classes=None, with_classes=False, pos=None):
        if self.latent_size > 2:
            if g_classes is not None or with_classes or pos is not None:
                raise Exception("sample_highd does not support g_classes, with_classes, or pos arguemnts")
            return self.sample_highd(batch_size)

        if g_classes is None:
            g_classes = np.random.randint(self.n_classes, size=batch_size)

        if pos is None:
            pos = (g_classes + np.random.rand(batch_size))/ self.n_classes 
        zs = np.zeros((batch_size, 2))
        noises = np.random.randn(batch_size) / 50 * self.noise_level
        zs[pos < 1] = np.stack((np.ones_like(pos[pos < 1]) * -1 + noises[pos < 1], (pos[pos < 1] - 0.75) * 8 - 1), -1)
        zs[pos < 0.75] = np.stack(((2 - (pos[pos < 0.75] - 0.5) * 8) - 1, np.ones_like(pos[pos < 0.75]) * -1 + noises[pos < 0.75]), -1)
        zs[pos < 0.5] = np.stack((np.ones_like(pos[pos < 0.5]) * 1 + noises[pos < 0.5], (2 - (pos[pos < 0.5] - 0.25) * 8) - 1), -1)
        zs[pos < 0.25] = np.stack((pos[pos < 0.25] * 8 - 1, np.ones_like(pos[pos < 0.25]) * 1 + noises[pos < 0.25]), -1)
        
        zs = zs * self.mult

        if with_classes:
            return torch.Tensor(zs).to(self.device), torch.Tensor(g_classes).long().to(self.device)
        else:
            return torch.Tensor(zs).to(self.device)

    def sample_highd(self, batch_size, g_classes=None, with_classes=False, pos=None):
        noises = np.random.randn(batch_size, self.latent_size) / 50 * self.noise_level
        zs = torch.randint(0, 2, (batch_size, self.latent_size), dtype=torch.float64) * 2. - 1.
        zs[torch.arange(batch_size), torch.randint(0, self.latent_size, (batch_size,))] = torch.rand(batch_size, dtype=torch.float64) * 2 - 1
        zs = zs * self.mult
        zs = zs + noises
        return zs.float().to(self.device)


def sample_categorical(batch_size, n_clusters=10):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, n_clusters, batch_size)
    cat = np.eye(n_clusters)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat


def get_prior(config):
    if config.prior.type == 'gauss':
        return GaussPrior(config.model.latent_size, std=config.prior.gauss_std)
    elif config.prior.type == 'gauss_mix':
        return GaussMixPrior(config.model.latent_size, config.dataset.n_classes)
    elif config.prior.type == 'swiss_roll':
        return SwissRollPrior(config.model.latent_size, config.dataset.n_classes, noise_level=config.prior.noise_level, mult=config.prior.mult)
    elif config.prior.type == 'grid_gauss':
        return GridGaussPrior(config.model.latent_size, 30)
    elif config.prior.type == 'pin_wheel':
        return PinWheelPrior(config.model.latent_size, config.prior.n_arc, mult=config.prior.mult)
    elif config.prior.type == 'square':
        return SquarePrior(config.model.latent_size, config.dataset.n_classes, noise_level=config.prior.noise_level, mult=config.prior.mult)
    else:
        raise NotImplementedError