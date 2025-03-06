import logging
import numpy as np
import os
import random
import sys
import time
import torch
from collections import defaultdict
from torchvision.datasets import MNIST
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score, contingency_matrix, completeness_score, homogeneity_score, v_measure_score
from scipy.special import logsumexp

from algorithms.vaes import DiffVAE, ClusteringVAE, DiffVAE_semi_simple, AutoClusteringDiffVAE, ClusteringDiffVAE, IAF_VAE, VAE, IAFDiffVAE, IAF_VAE_semi_simple, VAE_semi_simple, VAE_semi, IAF_VAE_semi, DiffVAE_semi, ClusteringIAFVAE, H_IAF_VAE, H_VAE, VAEDiffusion, DiffVAEFull, DiffVAEBoth, DiffVAEWarmup, DiffVAEWarmup_semi, IWAE
from algorithms.aaes import AAE_vanilla, AAE_semi, AAE_w_cluster_heads
from algorithms.baselines import PCAModel, TSNEModel, UMAPModel
from data.cifar10 import get_cifar10, get_cifar10_labels
from data.mnist import get_mnist, get_mnist_labels
from data.modern_eurasia import get_modern_eurasia, get_modern_eurasia_labels
from data.onekgenome import get_1kgenome, get_1kgenome_labels
from data.toy import get_toy_data
from priors import GridGaussPrior, get_prior
from utils import visualize_latent, visualize_1kgenome, visualize_eurasia, visualize_mnist, mmd_loss, MMDLoss

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


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    mat = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(mat, axis=0)) / np.sum(mat) 


def fit_and_score(fit_latents, score_latents):
    potential_sigmas = np.asarray([0.05, 0.8, 0.1, 0.3, 0.5])
    log.info(potential_sigmas)
    yo = []
    for sigma in potential_sigmas:
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(fit_latents) # atol=0.0005,rtol=0.01

        log_scores = kde.score_samples(score_latents)
        yo.append(log_scores)
    # log.info(yo)
    res = logsumexp(np.asarray(yo), 0) - np.log(len(potential_sigmas))
    return np.mean(res)
    

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


def get_data(config):
    if config.dataset.name == 'mnist':
        train_dataloader = get_mnist(config.model.batch_size, True, flattening=True, labels=False)
        test_dataloader = get_mnist(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == 'mnist_semi':
        train_dataloader = get_mnist(config.model.batch_size, True, flattening=True, labels=True, n_labels=config.dataset.n_labels, seed=config.seed)
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
    elif config.dataset.name == 'cifar10_semi':
        train_dataloader = get_cifar10(config.model.batch_size, True, flattening=False, labels=True, n_labels=config.dataset.n_labels, seed=config.seed)
        test_dataloader = get_cifar10(config.model.batch_size, False, flattening=False, labels=False)
    elif config.dataset.name == 'modern_eurasia':
        train_dataloader = get_modern_eurasia(config.model.batch_size, True, flattening=True, labels=False)
        test_dataloader = get_modern_eurasia(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == 'modern_eurasia_semi':
        train_dataloader = get_modern_eurasia(config.model.batch_size, True, flattening=True, labels=True)
        test_dataloader = get_modern_eurasia(config.model.batch_size, False, flattening=True, labels=False)
    elif config.dataset.name == '1kgenome':
        train_dataloader = get_1kgenome(config.model.batch_size, True, flattening=True, labels=False, small_pcs=config.dataset.small)
        test_dataloader = get_1kgenome(config.model.batch_size, False, flattening=True, labels=False, small_pcs=config.dataset.small)
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
    elif config.dataset.name in ['cifar10', 'cifar10_semi']:
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
        train_dataloader = get_1kgenome(config.model.batch_size, True, flattening=True, labels=False, shuffle=False, small_pcs=config.dataset.small)
        test_dataloader = get_1kgenome(config.model.batch_size, False, flattening=True, labels=False, shuffle=False, small_pcs=config.dataset.small)
        train_labels = get_1kgenome_labels(True)
        test_labels = get_1kgenome_labels(False)
    else:
        raise NotImplementedError
    return train_dataloader, test_dataloader, train_labels, test_labels


def metrics_evaluate(model_class):
    config = model_class.config
    train_dataloader, test_dataloader, train_labels, test_labels = get_eval_data(config)

    eval_output = model_class.eval(test_dataloader)
    if isinstance(eval_output, tuple):
        latents, ys = eval_output
    else:
        latents = eval_output

    with open(os.path.join(config.save_folder, 'latent.npy'), 'wb') as f:
        np.save(f, latents)

    if config.dataset.name.startswith('1kgenome'):
        visualize_1kgenome(latents[:, :2], os.path.join(config.save_folder, f'latent_z_final.png'))
    elif config.dataset.name.startswith('modern_eurasia'):
        visualize_eurasia(latents[:, :2], os.path.join(config.save_folder, f'latent_z_final.png'))
    else:
        visualize_latent(latents[:, :2], os.path.join(config.save_folder, f'latent_z_final.png'), targets=test_labels)

    res = []

    # classification accuracy
    train_output = model_class.eval(train_dataloader)
    if isinstance(train_output, tuple):
        train_latents, _ = train_output
    else:
        train_latents = train_output
    for nn in [20]:
        clf = KNeighborsClassifier(n_neighbors=nn).fit(train_latents, train_labels)
        sc = clf.score(latents, test_labels)
        log.info(f'KNN acc with {nn} neighbors: {sc}')
        if nn == 20:
            res.append(sc)
    
    # Latents LL
    prior = get_prior(config)
    prior_samples = prior.sample(10000).cpu().numpy()
    e_p_and_q = fit_and_score(latents, prior_samples)
    log.info(f'E_p [-log q(x)] = {-e_p_and_q}')
    res = res + [-e_p_and_q]
    
    # if config.model.name in ['aae_dim', 'diff_vae_clustering', 'diff_vae_autoclustering', 'clustering_vae', 'pca', 'tsne', 'umap', 'clustering_iaf_vae']:
    #     predicted_clusters = model_class.eval_label(test_dataloader)

    #     with open(os.path.join(config.save_folder, 'clusters.npy'), 'wb') as f:
    #         np.save(f, predicted_clusters)

    #     # Cluster purity
    #     cluster_purity = purity_score(test_labels, predicted_clusters)
    #     log.info(f'Cluster purity = {cluster_purity}')
    #     res = res + [cluster_purity]

    #     # Cluster completeness
    #     cluster_completeness = completeness_score(test_labels, predicted_clusters)
    #     log.info(f'Cluster completeness = {cluster_completeness}')
    #     res = res + [cluster_completeness]
        
    #     # NMI
    #     nmi = normalized_mutual_info_score(test_labels, predicted_clusters)
    #     log.info(f'NMI = {nmi}')
    #     res = res + [nmi]
    prior_samples = model_class.prior.sample(10000)
    sampled_imgs = []
    model_class.P.eval()
    for i in range(0, 10000, model_class.batch_size):
        with torch.no_grad():
            imgs = model_class.P(prior_samples[i:i+model_class.batch_size])
            # sampled_imgs.append(imgs.cpu().numpy())
            sampled_imgs.append(imgs.cpu())

    # sampled_imgs = np.concatenate(sampled_imgs, 0)
    sampled_imgs = torch.cat(sampled_imgs, 0)

    test_imgs = []
    for x in test_dataloader:
        test_imgs.append(x.cpu())
    test_imgs = torch.cat(test_imgs, 0)

    if len(sampled_imgs.size()) > 2: 
        sampled_imgs = sampled_imgs.view(10000, -1)
        test_imgs = test_imgs.view(10000, -1)
    res.append(mmd_loss(test_imgs, sampled_imgs))
    log.info(f'MMD now: {res[-1]}')

    try:
        res.append(model_class.get_elbo(test_dataloader)) # ELBO
    except:
        res.append(0)

    log.info(f'ELBO now: {res[-1]}')

    res = [time.time() - model_class.start_time] + res

    if os.path.exists(os.path.join(config.save_folder, 'res.npy')):
        with open(os.path.join(config.save_folder, 'res.npy'), 'rb') as f:
            old_res = np.load(f)
        new_res = np.concatenate((old_res, np.asarray(res)[None,:]))
    else:
        new_res = np.asarray(res)[None,:]
        
    with open(os.path.join(config.save_folder, 'res.npy'), 'wb') as f:
        np.save(f, new_res)


def training_routine(save_folder, model_class, epoch, max_epoch):
    config = model_class.config
    _, test_dataloader, _, test_labels = get_eval_data(config)

    if test_dataloader is None or (epoch % 5 != 0 and epoch != max_epoch - 1):
        return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    st = time.time()
    eval_output = model_class.eval(test_dataloader)
    log.info(f'Inference time = {(time.time()- st)/len(test_dataloader)}')
    if isinstance(eval_output, tuple):
        latents, ys = eval_output

        if config.dataset.name.startswith('1kgenome'):
            visualize_1kgenome(ys[:, :2], os.path.join(save_folder, f'latent_y_ep{epoch}.png'))
        elif config.dataset.name.startswith('modern_eurasia'):
            visualize_eurasia(ys[:, :2], os.path.join(save_folder, f'latent_y_ep{epoch}.png'))
        else:
            visualize_latent(ys[:, :2], os.path.join(save_folder, f'latent_y_ep{epoch}.png'), targets=test_labels)
    else:
        latents = eval_output

    if config.dataset.name.startswith('1kgenome'):
        visualize_1kgenome(latents[:, :2], os.path.join(save_folder, f'latent_z_ep{epoch}.png'))
    elif config.dataset.name.startswith('modern_eurasia'):
        visualize_eurasia(latents[:, :2], os.path.join(save_folder, f'latent_z_ep{epoch}.png'))
    else:
        visualize_latent(latents[:, :2], os.path.join(save_folder, f'latent_z_ep{epoch}.png'), targets=test_labels)

    # if epoch == max_epoch - 1:
    metrics_evaluate(model_class)


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

    log.info(f'Slurm job id {os.environ["SLURM_JOB_ID"]}')

    log.info(f'Loaded config: \n {OmegaConf.to_yaml(config)}')

    model = get_model(config)

    train_dataloader, test_dataloader = get_data(config)

    if config.model.name in ['pca', 'tsne', 'umap']:
        metrics_evaluate(model)
    else:
        if config.get_elbo:
            model.load(os.path.join(config.save_folder, 'model.pt'))

            res = []

            train_dataloader, test_dataloader, train_labels, test_labels = get_eval_data(config)

            eval_output = model.eval(test_dataloader)
            if isinstance(eval_output, tuple):
                latents, ys = eval_output
            else:
                latents = eval_output

            # classification accuracy
            train_output = model.eval(train_dataloader)
            if isinstance(train_output, tuple):
                train_latents, _ = train_output
            else:
                train_latents = train_output
            for nn in [20]:
                clf = KNeighborsClassifier(n_neighbors=nn).fit(train_latents, train_labels)
                sc = clf.score(latents, test_labels)
                log.info(f'KNN acc with {nn} neighbors: {sc}')
                if nn == 20:
                    res.append(sc)
            
            # Latents LL
            prior = get_prior(config)
            prior_samples = prior.sample(10000).cpu().numpy()
            e_p_and_q = fit_and_score(latents, prior_samples)
            log.info(f'E_p [-log q(x)] = {-e_p_and_q}')
            res = res + [-e_p_and_q]

            prior = get_prior(config)
            prior_samples = prior.sample(10000)
            sampled_imgs = []
            model.P.eval()
            for i in range(0, 10000, config.model.batch_size):
                with torch.no_grad():
                    imgs = model.P(prior_samples[i:i+config.model.batch_size])
                    # sampled_imgs.append(imgs.cpu().numpy())
                    sampled_imgs.append(imgs.cpu())
                # if i==0:
                #     visualize_mnist(imgs.cpu(), os.path.join(config.save_folder, f'look.png'))

            # sampled_imgs = np.concatenate(sampled_imgs, 0)
            sampled_imgs = torch.cat(sampled_imgs, 0)

            test_imgs = []
            _, test_dataloader = get_data(config)
            ct = 0
            for x in test_dataloader:
                # for img in x:
                    # test_imgs.append(imgs.cpu().numpy())
                test_imgs.append(x.cpu())
            # test_imgs = np.concatenate(test_imgs, 0)
            test_imgs = torch.cat(test_imgs, 0)

            # indices = np.random.choice(10000, 1000, replace=False)

            # elbo = [fit_and_score(sampled_imgs[indices], test_imgs[indices])]
            # log.info(f'NLL: {elbo[0]}')

            # crit = MMDLoss()
            # elbo = [crit(test_imgs[indices], sampled_imgs[indices])]
            # log.info(f'MMD RBF: {elbo[0]}')
            if len(sampled_imgs.size()) > 2: 
                sampled_imgs = sampled_imgs.view(10000, -1)
                test_imgs = test_imgs.view(10000, -1)
            res.append(mmd_loss(test_imgs, sampled_imgs))
            log.info(f'MMD: {res[-1]}')

            try:
                res.append(model.get_elbo(test_dataloader))
                log.info(f'ELBO: {res[-1]}')
            except:
                res.append(0)

            new_elbo = np.asarray(res)[None,:]
                
            with open(os.path.join(config.save_folder, 'elbo.npy'), 'wb') as f:
                np.save(f, new_elbo)
        elif not config.load_model:
            model.train(train_dataloader, training_routine=training_routine)
            # log.info(f'ELBO: {model.get_elbo(test_dataloader)}')
            model.save(os.path.join(config.save_folder, 'model.pt'))
        else:
            model.load(os.path.join(config.save_folder, 'model.pt'))
            metrics_evaluate(model)
            log.info(f'ELBO: {model.get_elbo(test_dataloader)}')


if __name__ == '__main__':
    main()