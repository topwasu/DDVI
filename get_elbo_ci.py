import logging
import os
import numpy as np
import scipy.stats
import sys

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
    for run in range(3): 
        with open(os.path.join(config.save_folder, f'run{run}', 'elbo.npy'), 'rb') as f:
            run_res = np.load(f)
        log.info(f'ELBO Run {run}: {run_res} {run_res.shape}')
        all_res.append(run_res)
    real_len = len(all_res[-1])
    all_res[0] = all_res[0][-real_len:]
    all_res = np.asarray(all_res)

    for i in range(all_res.shape[1]):
        m, h = mean_confidence_interval(all_res[:, i])
        log.info(f'Idx {i}: {m} +/- {h}')


if __name__ == '__main__':
    main()
