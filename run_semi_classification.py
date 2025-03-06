import logging
import numpy as np
import sys
import torch
from collections import defaultdict
from torchvision.datasets import MNIST

from priors import GridGaussPrior
from algorithms.vaes import SemiDiffVAE, SemiAutoClusteringVAE
from data.mnist import MNISTConfig, get_mnist

logger_ = logging.getLogger()
logger_.level = logging.INFO # important

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
stream_handler.setFormatter(formatter)

logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)


def unsupervised_score(latents):
    # prior = GridGaussPrior(10, 10)
    prior = GridGaussPrior(2, 10)
    cluster_probs = prior.evaluate_class(torch.from_numpy(latents)).numpy()
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
    for i in range(10):
        try:
            cluster_label = i
            err_ct += len(np.nonzero(targets[latent_ids_by_cluster[i]] != cluster_label)[0])
        except:
            continue
    return err_ct / 10000 * 100


def main():
    batch_size = 128
    config = MNISTConfig
    train_dataloader = get_mnist(batch_size, True, flattening=True, labels=True, n_labels=100)
    test_dataloader = get_mnist(batch_size, False, flattening=True, labels=False)

    config.n_clusters = 10
    # config.latent_size = 10
    # config.prior = GridGaussPrior(10, config.n_clusters)
    config.latent_size = 2
    config.prior = GridGaussPrior(2, config.n_clusters)
    config.batch_size = 128
    config.lr = 0.0001
    # config.lr = 0.0005
    config.timesteps = 10
    config.num_epochs = 400
    config.hidden_size = 1000
    config.kld_weight = 0.01
    config.kld_schedule = 'constant'
    config.kld_warmup = False
    config.criterion = 'bce'
    config.g_path = None
    # config.g_path = 'standard_grid_gauss_lr0.0005_bs128_t10_ep500_latent5/model.pt'
    config.save_folder = 'semi_classification'

    model = SemiDiffVAE(config)
    # model = SemiAutoClusteringVAE(config)

    log.info(f'Save folder {config.save_folder}')

    model.train(train_dataloader, test_dataloader, eval_func=lambda x: unsupervised_score(x, config.n_clusters))

if __name__ == '__main__':
    main()
