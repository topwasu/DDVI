import logging
import numpy as np
import os
import shutil
import random
import scipy.stats
import sys
import torch
import cv2
from cleanfid import fid
from PIL import Image
from collections import defaultdict
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score, contingency_matrix
from scipy.special import logsumexp

from algorithms.vaes import *
from algorithms.aaes import *
from data.cifar10 import get_cifar10, get_cifar10_labels
from data.mnist import get_mnist, get_mnist_labels
from data.modern_eurasia import get_modern_eurasia, get_modern_eurasia_labels
from data.onekgenome import get_1kgenome, get_1kgenome_labels
from data.toy import get_toy_data
from priors import GridGaussPrior, get_prior, SquarePrior
from utils import visualize_latent, visualize_1kgenome, visualize_eurasia, visualize_mnist

import hydra
from omegaconf import OmegaConf


logger_ = logging.getLogger()
logger_.level = logging.INFO # important

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
stream_handler.setFormatter(formatter)

logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.asarray(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a, 0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def fit_and_score(fit_latents, score_latents, sigmas=[0.005, 0.008, 0.01, 0.03, 0.05]): # TODO
    potential_sigmas = np.asarray(sigmas)
    log.info(potential_sigmas)
    yo = []
    for sigma in potential_sigmas:
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(fit_latents)

        log_scores = kde.score_samples(score_latents)
        yo.append(log_scores)
    res = logsumexp(np.asarray(yo), 0) - np.log(len(potential_sigmas))
    return np.mean(res)


def get_data(config):
    if config.dataset.name == 'mnist':
        train_dataloader = get_mnist(config.model.batch_size, True, flattening=True, labels=False)
        test_dataloader = get_mnist(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == 'mnist_semi':
        train_dataloader = get_mnist(config.model.batch_size, True, flattening=True, labels=True, n_labels=config.dataset.n_labels)
        test_dataloader = get_mnist(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == 'mnist_toy':
        train_dataloader = get_mnist(config.model.batch_size, True, flattening=True, labels=False, toy=True)
        test_dataloader = train_dataloader
    elif config.dataset.name.startswith('toy'):
        train_dataloader = get_toy_data(config.dataset.name, config.model.batch_size, True)
        test_dataloader = get_toy_data(config.dataset.name, config.model.batch_size, False)
    elif config.dataset.name == 'cifar10':
        train_dataloader = get_cifar10(config.model.batch_size, True, flattening=False, labels=False)
        test_dataloader = get_cifar10(config.model.batch_size, False, flattening=False, labels=False)
    elif config.dataset.name == 'modern_eurasia':
        train_dataloader = get_modern_eurasia(config.model.batch_size, True, flattening=True, labels=False)
        test_dataloader = get_modern_eurasia(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == 'modern_eurasia_semi':
        train_dataloader = get_modern_eurasia(config.model.batch_size, True, flattening=True, labels=True)
        test_dataloader = get_modern_eurasia(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == '1kgenome':
        train_dataloader = get_1kgenome(config.model.batch_size, True, flattening=True, labels=False)
        test_dataloader = get_1kgenome(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == '1kgenome_semi':
        train_dataloader = get_1kgenome(config.model.batch_size, True, flattening=True, labels=True)
        test_dataloader = get_1kgenome(config.model.batch_size, False, flattening=True, labels=False)
    else:
        raise NotImplementedError
    return train_dataloader, test_dataloader


def get_eval_data(config):
    if config.dataset.name in ['mnist', 'mnist_semi']:
        train_dataloader = get_mnist(config.model.batch_size, True, flattening=True, labels=False, shuffle=False)
        test_dataloader = get_mnist(config.model.batch_size, False, flattening=True, labels=False, shuffle=False)
        train_labels = get_mnist_labels(True)
        test_labels = get_mnist_labels(False)
    elif config.dataset.name == 'mnist_toy':
        train_dataloader = get_mnist(config.model.batch_size, True, flattening=True, labels=False, shuffle=False, toy=True)
        test_dataloader = train_dataloader
        train_labels = get_mnist_labels(True, toy=True)
        test_labels = train_labels
    elif config.dataset.name == 'cifar10':
        train_dataloader = get_cifar10(config.model.batch_size, True, flattening=False, labels=False, shuffle=False)
        test_dataloader = get_cifar10(config.model.batch_size, False, flattening=False, labels=False, shuffle=False)
        train_labels = get_cifar10_labels(True)
        test_labels = get_cifar10_labels(False)
    elif config.dataset.name in ['modern_eurasia', 'modern_eurasia_semi']:
        train_dataloader = get_modern_eurasia(config.model.batch_size, True, flattening=True, labels=False, shuffle=False)
        test_dataloader = get_modern_eurasia(config.model.batch_size, False, flattening=True, labels=False, shuffle=False)
        train_labels = get_modern_eurasia_labels(True)
        test_labels = get_modern_eurasia_labels(False)
    elif config.dataset.name in ['1kgenome', '1kgenome_semi']:
        train_dataloader = get_1kgenome(config.model.batch_size, True, flattening=True, labels=False, shuffle=False)
        test_dataloader = get_1kgenome(config.model.batch_size, False, flattening=True, labels=False, shuffle=False)
        train_labels = get_1kgenome_labels(True)
        test_labels = get_1kgenome_labels(False)
    else:
        raise NotImplementedError
    return train_dataloader, test_dataloader, train_labels, test_labels


def get_model(config):
    if config.model.name == 'diff_vae':
        return DiffVAE(config)
    elif config.model.name == 'iaf_diff_vae':
        return IAFDiffVAE(config)
    elif config.model.name == 'diff_vae_semi_simple':
        return DiffVAE_semi_simple(config)
    elif config.model.name == 'diff_vae_semi':
        return DiffVAE_semi(config)
    elif config.model.name == 'diff_vae_autoclustering':
        return AutoClusteringDiffVAE(config)
    elif config.model.name == 'diff_vae_clustering':
        return ClusteringDiffVAE(config)
    elif config.model.name == 'clustering_vae':
        return ClusteringVAE(config)
    elif config.model.name == 'clustering_iaf_vae':
        return ClusteringIAFVAE(config)
    elif config.model.name == 'aae_vanilla':
        return AAE_vanilla(config)
    elif config.model.name == 'aae_semi':
        return AAE_semi(config)
    elif config.model.name == 'aae_dim':
        return AAE_w_cluster_heads(config)
    elif config.model.name == 'iaf_vae':
        return IAF_VAE(config)
    elif config.model.name == 'vae':
        return VAE(config)
    elif config.model.name == 'iwae':
        return IWAE(config)
    elif config.model.name == 'iaf_vae_semi_simple':
        return IAF_VAE_semi_simple(config)
    elif config.model.name == 'vae_semi_simple':
        return VAE_semi_simple(config)
    elif config.model.name == 'iaf_vae_semi':
        return IAF_VAE_semi(config)
    elif config.model.name == 'vae_semi':
        return VAE_semi(config)
    elif config.model.name == 'pca':
        return PCAModel(config)
    elif config.model.name == 'tsne':
        return TSNEModel(config)
    elif config.model.name == 'umap':
        return UMAPModel(config)
    elif config.model.name == 'h_vae':
        return H_VAE(config)
    elif config.model.name == 'h_iaf_vae':
        return H_IAF_VAE(config)
    elif config.model.name == 'vae_diff':
        return VAEDiffusion(config)
    elif config.model.name == 'diff_vae_full':
        return DiffVAEFull(config)
    elif config.model.name == 'diff_vae_both':
        return DiffVAEBoth(config)
    elif config.model.name == 'diff_vae_warmup':
        return DiffVAEWarmup(config)
    elif config.model.name == 'diff_vae_warmup_semi':
        return DiffVAEWarmup_semi(config)
    else:
        raise NotImplementedError
    

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    logger_ = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(config.save_folder, 'run.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger_.addHandler(file_handler)

    set_seed(config.seed)

    log.info("THE WALK")

    # log.info(f'Slurm job id {os.environ["SLURM_JOB_ID"]}')

    # log.info(f'Loaded config: \n {OmegaConf.to_yaml(config)}')

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    all_res = []
    # for seed in range(5):
        # config.g_path = os.path.join(config.save_folder, f'run{seed}', 'diff_model.pt')
    # config.g_path = os.path.join(config.save_folder, 'diff_model.pt')

    model = get_model(config)

    model.load(os.path.join(config.save_folder, f'run2', 'model.pt'))
    # model.load(os.path.join(config.save_folder, 'model.pt'))
    
    dir_base = os.path.join(config.save_folder)
    prior = get_prior(config)
    for j in range(1):
        imgs = []
        np_samples = []
        for i in range(100):
            sample = prior.sample(1)
            with torch.no_grad():
                img = model.P(sample)
                imgs.append(img.cpu())
            np_samples.append(sample.cpu().numpy())
        imgs = torch.cat(imgs, 0)
        visualize_mnist(imgs, os.path.join(config.save_folder, f'imgs{j}.png'))
        for idx, sample in enumerate(np_samples):
            log.info(f'Idx {idx}: {list(sample)}')
        # dir_base = os.path.join(config.save_folder, f'run{seed}')
        # dir_predicted = os.path.join(config.save_folder, f'run{seed}', 'predicted')

        # if os.path.exists(dir_predicted) and os.path.isdir(dir_predicted):
        #     shutil.rmtree(dir_predicted)
        # os.makedirs(dir_predicted, exist_ok=True)

        # dir_test = os.path.join(config.save_folder, f'run{seed}', 'test')

        # if os.path.exists(dir_test) and os.path.isdir(dir_test):
        #     shutil.rmtree(dir_test)
        # os.makedirs(dir_test, exist_ok=True)
        # prior_samples = prior.sample(10000)
        # all_imgs = []
        # ct = 0
        # for i in range(0, 10000, config.model.batch_size):
        #     with torch.no_grad():
        #         imgs = model.P(prior_samples[i:i+config.model.batch_size])
        #     for img in imgs:
        #         if ct >= 10:
        #             break
        #         save_image(img.reshape(1, 28, 28), os.path.join(dir_base, f"{ct}.png"))
        #         # save_image(img.reshape(1, 28, 28), os.path.join(dir_predicted, f"{ct}.png"))
        #         ct += 1
        #     break
        #     all_imgs.append(imgs.cpu().numpy())
        # all_imgs = np.concatenate(all_imgs, axis=0)

    #     _, test_dataloader = get_data(config)
    #     all_test_imgs = []
    #     ct = 0
    #     for x in test_dataloader:
    #         for img in x:
    #             if ct >= 10:
    #                 break
    #             save_image(img.reshape(1, 28, 28), os.path.join(dir_test, f"{ct}.png"))
    #             ct += 1
    #         break
    #     #     all_test_imgs.append(x.cpu().numpy())
    #     # all_test_imgs = np.concatenate(all_test_imgs, axis=0)
        

    #     # all_imgs = np.asarray(all_imgs, dtype=np.uint8)
    #     # all_imgs = all_imgs.reshape((-1,28,28))
    #     # all_test_imgs = np.asarray(all_test_imgs, dtype=np.uint8)
    #     # all_test_imgs = all_test_imgs.reshape((-1,28,28))

    #     # dir_predicted = os.path.join(config.save_folder, f'run{seed}', 'predicted')
    #     # os.makedirs(dir_predicted, exist_ok=True)
    #     # for idx, img in enumerate(all_imgs):
    #     #     cv2.imwrite(os.path.join(dir_predicted, f"{idx}.jpeg"), img)

    #     # dir_test = os.path.join(config.save_folder, f'run{seed}', 'test')
    #     # os.makedirs(dir_test, exist_ok=True)
    #     # for idx, img in enumerate(all_test_imgs):
    #     #     cv2.imwrite(os.path.join(dir_test, f"{idx}.jpeg"), img)


    #     run_res = [fid.compute_fid(dir_predicted, dir_test)]
    #     log.info(f'Seed {seed}: {run_res}')
    #     all_res.append(run_res)
    # all_res = np.asarray(all_res)

    # for i in range(all_res.shape[1]):
    #     m, h = mean_confidence_interval(all_res[:, i])
    #     log.info(f'Idx {i}: {m} +/- {h}')


if __name__ == '__main__':
    main()