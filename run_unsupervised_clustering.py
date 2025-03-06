import logging
import numpy as np
import sys
import torch
from collections import defaultdict
from torchvision.datasets import MNIST

from priors import GridGaussPrior
from algorithms.vaes import DiffVAE, AutoClusteringVAE, DirichletVAE, NewDirichletVAE, MMDVAE, AutoClusteringDiffVAE, HybridVAE, YVAE, DiscreteVAE
from data.mnist import MNISTConfig, get_mnist

logger_ = logging.getLogger()
logger_.level = logging.INFO # important

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
stream_handler.setFormatter(formatter)

logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)


def unsupervised_score(inputs, n_clusters, input='latents'):
    if input == 'latents':
        latents = inputs
        prior = GridGaussPrior(5, n_clusters)
        cluster_probs = prior.evaluate_class(torch.from_numpy(latents)).numpy()
    elif input == 'cluster_probs':
        cluster_probs = inputs
    else:
        raise NotImplementedError

    latent_clusters = np.argmax(cluster_probs, -1)

    latent_ids_by_cluster = defaultdict(list)
    best_id_by_cluster = {}
    for id, latent_cluster in enumerate(latent_clusters):
        latent_ids_by_cluster[latent_cluster].append(id)
        if (latent_cluster not in best_id_by_cluster) or (cluster_probs[best_id_by_cluster[latent_cluster]][latent_cluster] < cluster_probs[id][latent_cluster]):
            best_id_by_cluster[latent_cluster] = id

    err_ct = 0
    test_dataset = MNIST('.', train=False, download=True)
    targets = test_dataset.targets.numpy()
    for i in range(n_clusters):
        try:
            cluster_label = targets[best_id_by_cluster[i]]
            err_ct += len(np.nonzero(targets[latent_ids_by_cluster[i]] != cluster_label)[0])
        except:
            continue
    return err_ct / 10000 * 100


def main():
    batch_size = 128
    config = MNISTConfig
    train_dataloader = get_mnist(batch_size, True, flattening=True, labels=False)
    test_dataloader = get_mnist(batch_size, False, flattening=True, labels=False)

    config.gauss_std = 1
    # config.n_clusters = 30
    config.n_clusters = 20
    # config.latent_size = 5
    # config.prior = GridGaussPrior(5, config.n_clusters)
    config.latent_size = 2
    config.prior = GridGaussPrior(2, config.n_clusters)
    config.batch_size = 128
    # config.lr = 0.001
    config.lr = 0.0001
    config.timesteps = 10
    config.num_epochs = 400
    # config.hidden_size = 300
    config.hidden_size = 1000
    config.kld_weight = 0.01
    config.kld_schedule = 'linear'
    config.kld_warmup = False
    config.criterion = 'bce'
    # config.g_path = None
    # config.g_path = 'standard_grid_gauss_lr0.0005_bs128_t10_ep500_latent5/model.pt'
    # config.save_folder = None
    # config.save_folder = 'unsupervised_2d'
    # config.save_folder = 'unsupervised_5d'
    # config.save_folder = 'unsupervised_2d_dirichletvae_20clusters_lr0.001_dirreg1kld_std0.3_kld0.005linear_pre_exp_fixed'
    config.save_folder = 'unsupervised_2d_dirichletvae_20clusters_newalpha_newkl_lin_ep400_ent0.01_std1'
    # config.save_folder = 'unsupervised_2d_yvae_20clusters_ep400_kld0.01_std0.3_pre'
    # config.save_folder = 'unsupervised_2d_discretevae_20clusters_ep100_kld0.01con_std0.3_pre_temp0.2'

    # model = DiffVAE(config)
    # model = AutoClusteringVAE(config)
    # model = DirichletVAE(config)
    model = NewDirichletVAE(config)
    # model = MMDVAE(config)
    # model = AutoClusteringDiffVAE(config)
    # model = HybridVAE(config)
    # model = YVAE(config)
    # model = DiscreteVAE(config)

    log.info(f'Save folder {config.save_folder}')

    # model.train(train_dataloader, test_dataloader, eval_func=lambda x: unsupervised_score(x, config.n_clusters))
    # model.train(train_dataloader, test_dataloader, eval_func=lambda x: unsupervised_score(x, config.n_clusters, 'cluster_probs'))
    model.train(train_dataloader, test_dataloader)

if __name__ == '__main__':
    main()
