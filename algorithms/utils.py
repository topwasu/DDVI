import numpy as np  
import torch
import logging 
from models.simple import MultiWayLinear
from models.transposed_conv_net import ConvNet, TransposedConvNet, BareConvNet
from models.hvae import Q2


log = logging.getLogger(__name__)


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def get_kld_weights(kld_weight, num_epochs, kld_schedule, kld_warmup=False):
    if kld_schedule == 'constant':
        kld_weights = frange_cycle_linear(num_epochs, start=kld_weight, stop=kld_weight)
    elif kld_schedule == 'frequent_cyclic':
        kld_weights = np.tile(np.asarray([kld_weight] * 5 + [0] * 5), num_epochs // 10 + 1)[:num_epochs]
        kld_weights[-20:] = kld_weight
    elif kld_schedule == 'reverse_warmup':
        kld_weights = np.zeros(num_epochs)
        kld_weights[5:] = kld_weight
    else:
        warmup_length = int(num_epochs * 0.1) if kld_warmup else 0
        schedule_length = int(num_epochs * 0.7) if kld_warmup else int(num_epochs * 0.8)
        if kld_schedule == 'linear':
            kld_weights = frange_cycle_linear(schedule_length, start=0, stop=kld_weight, n_cycle=1, ratio=1)
        elif kld_schedule == 'cyclic':
            kld_weights = frange_cycle_linear(schedule_length, start=0, stop=kld_weight, n_cycle=4, ratio=0.8)
        kld_weights = np.concatenate((np.full(warmup_length, kld_weight), 
                                      kld_weights, 
                                      np.full(num_epochs - schedule_length - warmup_length, kld_weight)))
        
    return kld_weights


class AddNoise():
    def __init__(self, prior, sigma = 0):
        self.prior = prior
        self.sigma = sigma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def sample(self, batch_size, **kwargs):
        return self.prior.sample(batch_size, **kwargs) + torch.randn(batch_size, self.prior.latent_size, device=self.device) * self.sigma
    

def get_q_and_p(config):
    if config.model.name in ['aae_vanilla', 'aae_semi']:
        out_sizes = config.model.latent_size
    elif config.model.name == 'aae_dim':
        out_sizes = [config.model.n_clusters, config.model.latent_size]
    elif config.model.name in ['h_vae', 'vae', 'iwae', 'vae_semi_simple', 'diff_vae', 'diff_vae_semi_simple', 'diff_vae_autoclustering', 'diff_vae_warmup', 'vae_diff']:
        out_sizes = [config.model.latent_size, config.model.latent_size]
    elif config.model.name in ['vae_semi', 'diff_vae_semi', 'diff_vae_warmup_semi']:
        out_sizes = [config.model.latent_size, config.model.latent_size, config.dataset.n_classes]
    elif config.model.name in ['clustering_vae', 'diff_vae_clustering']:
        out_sizes = [config.model.latent_size, config.model.latent_size, config.model.n_clusters, config.model.n_clusters]
    elif config.model.name in ['h_iaf_vae', 'iaf_vae', 'iaf_vae_semi_simple', 'iaf_diff_vae', 'diff_vae_full', 'diff_vae_both']:
        out_sizes = [config.model.latent_size, config.model.latent_size, config.model.context_size]
    elif config.model.name in ['iaf_vae_semi']:
        out_sizes = [config.model.latent_size, config.model.latent_size, config.model.context_size, config.dataset.n_classes]
    elif config.model.name == 'clustering_iaf_vae':
        out_sizes = [config.model.latent_size, config.model.latent_size, config.model.n_clusters, config.model.n_clusters, config.model.context_size]
    else:
        log.info(f'Q and P not implemented for {config.model.name}')
        raise NotImplementedError

    if config.dataset.name.startswith('mnist'):
        Q = MultiWayLinear(784, [config.model.hidden_size, config.model.hidden_size], out_sizes, bn=config.model.bn, dropout=0.2)
        P = MultiWayLinear(config.model.latent_size, [config.model.hidden_size, config.model.hidden_size], 784, sigmoid=True, dropout=0.2)
    elif config.dataset.name.startswith('cifar'):
        fc = MultiWayLinear(128, [config.model.hidden_size], out_sizes, bn=config.model.bn, dropout=0.2)
        Q = ConvNet(config.model.latent_size, fc)
        P = TransposedConvNet(config.model.latent_size)
    elif config.dataset.name.startswith('modern_eurasia') or config.dataset.name.startswith('1kgenome'):
        if  config.dataset.name == '1kgenome' and config.dataset.small:
            Q = MultiWayLinear(15, [100, 100, 100], out_sizes, bn=config.model.bn, dropout=0.2)
            P = MultiWayLinear(config.model.latent_size, [100, 100, 100, 100], 15, dropout=0.2)
        else:
            Q = MultiWayLinear(1000, [config.model.hidden_size, config.model.hidden_size, config.model.hidden_size, config.model.hidden_size], out_sizes, bn=config.model.bn, dropout=0.2)
            P = MultiWayLinear(config.model.latent_size, [config.model.hidden_size, config.model.hidden_size, config.model.hidden_size, config.model.hidden_size], 1000, dropout=0.2)
    else:
        raise NotImplementedError
    
    return Q, P


def get_q2(config):
    if config.model.name in ['h_vae']:
        out_sizes = [config.model.latent_size, config.model.latent_size]
    elif config.model.name in ['h_iaf_vae']:
        out_sizes = [config.model.latent_size, config.model.latent_size, config.model.context_size]

    if config.dataset.name.startswith('mnist'):
        img_processor = MultiWayLinear(784, [config.model.hidden_size, config.model.hidden_size], 128, bn=config.model.bn, dropout=0.2)
    elif config.dataset.name.startswith('cifar'):
        img_processor = BareConvNet()
    
    fc = MultiWayLinear(128 + config.model.latent_size, [24, 24], out_sizes, bn=config.model.bn, dropout=0.2)
    q2 = Q2(img_processor, fc)

    return q2


def cs_entropy(cs):
    if len(cs) == 0:
        return torch.zeros(1).cuda()
    
    return -torch.sum((cs + 1e-6) * torch.log(cs + 1e-6)) / cs.size(0)


def gauss_logpdf_samevalue(std):
    return torch.mean(-torch.log(torch.tensor(2 * 3.14)) / 2 - torch.sum(torch.log(torch.tensor(std))))