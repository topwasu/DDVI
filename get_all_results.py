import logging
import os
import numpy as np
import scipy.stats
import sys
from sklearn.neighbors import KernelDensity
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score, contingency_matrix, completeness_score, homogeneity_score, v_measure_score

from priors import SquarePrior, SwissRollPrior, PinWheelPrior
from data.onekgenome import get_1kgenome, get_1kgenome_labels
from utils import visualize_cluster_latents

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


def get_prior(prior):
    if prior == 'noisy_swiss_roll':
        return SwissRollPrior(2, 10, noise_level=0.2, mult=1)
    elif prior == 'pin_wheel':
        return PinWheelPrior(2, 10, mult=1)
    elif prior == 'square':
        return SquarePrior(2, 10, noise_level=1, mult=1)
    elif prior == 'less_noisy_swiss_roll':
        return SwissRollPrior(2, 10, noise_level=0.1, mult=1)
    elif prior == 'less_noisy_square':
        return SquarePrior(2, 10, noise_level=3, mult=1)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    mat = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(mat, axis=0)) / np.sum(mat) 


def fit_and_score(fit_latents, score_latents):
    potential_sigmas = np.asarray([0.005, 0.008, 0.01, 0.03, 0.05])
    log.info(potential_sigmas)
    yo = []
    for sigma in potential_sigmas:
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(fit_latents)

        log_scores = kde.score_samples(score_latents)
        yo.append(log_scores)
    res = logsumexp(np.asarray(yo), 0) - np.log(len(potential_sigmas))
    return np.mean(res)
    

def metrics_evaluate(folder, prior, clustering=False):
    with open(os.path.join(folder, 'latent.npy'), 'rb') as f:
        latents = np.load(f)

    res = []
    if clustering:
        test_labels = get_1kgenome_labels(False) # TODO: Change
        log.info(len(test_labels))
        convert = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        new_test_labels = [convert[label] for label in test_labels]
        for nn in [20]:
            gmm = GaussianMixture(n_components=nn, random_state=2, covariance_type='full', max_iter=100, n_init=10, reg_covar=1e-2).fit(latents)
            predicted_clusters = gmm.predict(latents)
            visualize_cluster_latents(latents, os.path.join(folder, f'latents_{nn}clustered.png'), predicted_clusters)
            
            log.info(f'GMM ELBO {gmm.lower_bound_}')
            res = res + [gmm.lower_bound_]
            
            # Cluster purity
            cluster_purity = purity_score(test_labels, predicted_clusters)
            log.info(f'NN: {nn} - Cluster purity = {cluster_purity}')
            res = res + [cluster_purity]

            # Cluster completeness
            cluster_completeness = completeness_score(test_labels, predicted_clusters)
            log.info(f'NN: {nn} - Cluster completeness = {cluster_completeness}')
            res = res + [cluster_completeness]
            
            # NMI
            nmi = normalized_mutual_info_score(test_labels, predicted_clusters)
            log.info(f'NN: {nn} - NMI = {nmi}')
            res = res + [nmi]

            # Homogeneity
            hom = homogeneity_score(test_labels, predicted_clusters)
            log.info(f'NN: {nn} - Homogeneity = {hom}')
            res = res + [hom]

            # NMI
            v_score = v_measure_score(test_labels, predicted_clusters)
            log.info(f'NN: {nn} - v_score = {v_score}')
            res = res + [v_score]

            # # Continent Cluster purity
            # cluster_purity = purity_score(new_test_labels, predicted_clusters)
            # log.info(f'Continent NN: {nn} - Cluster purity = {cluster_purity}')
            # res = res + [cluster_purity]

            # # Continent Cluster completeness
            # cluster_completeness = completeness_score(new_test_labels, predicted_clusters)
            # log.info(f'Continent NN: {nn} - Cluster completeness = {cluster_completeness}')
            # res = res + [cluster_completeness]
            
            # # Continent NMI
            # nmi = normalized_mutual_info_score(new_test_labels, predicted_clusters)
            # log.info(f'Continent NN: {nn} - NMI = {nmi}')
            # res = res + [nmi]

            # # Continent Homogeneity
            # hom = homogeneity_score(new_test_labels, predicted_clusters)
            # log.info(f'Continent NN: {nn} - Homogeneity = {hom}')
            # res = res + [hom]

            # # Continent NMI
            # v_score = v_measure_score(new_test_labels, predicted_clusters)
            # log.info(f'Continent NN: {nn} - v_score = {v_score}')
            # res = res + [v_score]

            # predicted_clusters = GaussianMixture(n_components=nn, random_state=2, covariance_type='full', max_iter=100, n_init=10, reg_covar=1e-2).fit_predict(latents)
            # visualize_cluster_latents(latents, os.path.join(folder, f'latents_{nn}clustered.png'), predicted_clusters)

            # # for i in range(26):
            # #     clustersss = predicted_clusters[test_labels == i]
            # #     log.info(f'I {i}: bin {np.bincount(clustersss)}')

            # # Cluster purity
            # cluster_purity = purity_score(test_labels, predicted_clusters)
            # log.info(f'NN: {nn} - Cluster purity = {cluster_purity}')
            # res = res + [cluster_purity]

            # # Cluster completeness
            # cluster_completeness = completeness_score(test_labels, predicted_clusters)
            # log.info(f'NN: {nn} - Cluster completeness = {cluster_completeness}')
            # res = res + [cluster_completeness]
            
            # # NMI
            # nmi = normalized_mutual_info_score(test_labels, predicted_clusters)
            # log.info(f'NN: {nn} - NMI = {nmi}')
            # res = res + [nmi]

            # # Homogeneity
            # hom = homogeneity_score(test_labels, predicted_clusters)
            # log.info(f'NN: {nn} - Homogeneity = {hom}')
            # res = res + [hom]

            # # V Score
            # log.info(f'MIN {min(predicted_clusters)}') 
            # log.info(f'MAX {max(predicted_clusters)}')
            # v_score = v_measure_score(test_labels, predicted_clusters)
            # log.info(f'NN: {nn} - v_score = {v_score}')
            # res = res + [v_score]
        
        # for nn in [500, 1000, 2000]:
        #     clf = KNeighborsClassifier(n_neighbors=nn).fit(latents, new_test_labels)
        #     sc = clf.score(latents, new_test_labels)
        #     log.info(f'{nn} acc = {sc}')
        #     res = res + [sc]
        for nn in [20, 50, 100]:
            clf = KNeighborsClassifier(n_neighbors=nn).fit(latents, test_labels)
            sc = clf.score(latents, test_labels)
            log.info(f'{nn} acc = {sc}')
            res = res + [sc]
    else:
        # Latents LL
        prior = get_prior(prior)
        prior_samples = prior.sample(10000).cpu().numpy()
        e_p_and_q = fit_and_score(latents, prior_samples)
        # e_q_and_q = fit_and_score(latents, latents)
        # e_q_and_p = fit_and_score(prior_samples, latents)
        log.info(f'E_p [-log q(x)] = {-e_p_and_q}')
        # log.info(f'E_q [-log p(x)] = {-e_q_and_p}')
        # log.info(f'KL(q||p) = E_q [log q(x) - log p(x)] = {e_q_and_q - e_q_and_p}')
        res = [-e_p_and_q]
    
    return res


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.asarray(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a, 0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    logger_ = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(config.save_folder, 'run.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger_.addHandler(file_handler)

    # log.info(f'Slurm job id {os.environ["SLURM_JOB_ID"]}')

    log.info(f'Loaded config: \n {OmegaConf.to_yaml(config)}')

    all_res = []
    # for prior in ['pin_wheel', 'less_noisy_swiss_roll', 'less_noisy_square']:
    #     for method in ['iaf_vae', 'aae_vanilla', 'diff_vae']:
    for prior in ['tiny_gauss']:
        for method in ['pca', 'tsne', 'umap', 'aae_dim', 'clustering_vae_mse', 'diff_vae_autoclustering_mse']:
            log.info(f'Method {method} Prior {prior}')
            all_res = []
            for run in range(0, 5): 
                try:
                    # if method == 'diff_vae':
                    #     run_res = metrics_evaluate(f'final_results/mnist_{method}_{prior}_kld0.003/run{run}', prior)
                    # else:
                    #     run_res = metrics_evaluate(f'final_results/mnist_{method}_{prior}/run{run}', prior)
                    run_res = metrics_evaluate(f'final_results/onekgenome_{method}_{prior}/run{run}', prior, True)
                except Exception as e:
                    print(f'Exception: {e}')
                    break
                all_res.append(run_res)
            all_res = np.asarray(all_res)

            for i in range(all_res.shape[1]):
                m, h = mean_confidence_interval(all_res[:, i])
                log.info(f'Idx {i}: {m} +/- {h}')


if __name__ == '__main__':
    main()
