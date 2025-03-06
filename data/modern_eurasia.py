import logging
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import MNIST
from torchvision import transforms as tsfm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from .utils import CustomDataset


log = logging.getLogger(__name__)


def _get_modern_eurasia_labels(pop_name = 'all'):
    if pop_name not in ['all', 'west']:
        raise NotImplementedError

    folder = '/share/kuleshov/pop_gen_data/modern_eurasia'
    ind_df = pd.read_csv(f'{folder}/HumanOrigins_FennoScandian_small.ind', names=['sample_id', 'gender', 'status'], sep="\s+") 
    ind_df = ind_df.astype({'sample_id': 'str', 'gender': 'str', 'status': 'str'})
    snp_df = pd.read_csv(f'{folder}/HumanOrigins_FennoScandian_small.snp', names=['snp_id', 'chromosome_num', 'genetic_position', 'physical_position', 'ref_allele', 'alternative_allele'], sep="\s+") 
    snp_df = snp_df.astype({'snp_id' : 'int', 'chromosome_num': 'int', 'genetic_position': 'str', 'physical_position': 'int', 'ref_allele': 'str', 'alternative_allele': 'str'})
    geno_df = pd.read_csv(f'{folder}/HumanOrigins_FennoScandian_small.geno', names=['genotype'], sep="\s+") 
    geno_df = geno_df.astype({'genotype': 'str'})

    
    filename = 'WestEurasia.poplist.txt' if pop_name == 'west' else 'AllEurasia.poplist.txt'
    with open(f'{folder}/{filename}') as f:
        lines = f.readlines()
    pop = [line.strip() for line in lines]
    
    # choose population
    log.info("Filtering data accord to interested population")
    status_arr = ind_df['status'].to_numpy()
    pop_indices = np.where(np.isin(status_arr, pop))[0]
    targets = ind_df['status'].iloc[pop_indices]

    numbered_targets = [pop.index(target) for target in targets.to_list()]
    return numbered_targets


def get_modern_eurasia_pcs():
    with open('/share/kuleshov/pop_gen_data/modern_eurasia/eurasia_1000pcs.npy', 'rb') as f:
        pca_pcs = np.load(f)
    # pca_pcs = pca_pcs[:, :1000]
    # data = (pca_pcs - np.mean(pca_pcs, 0)) / np.std(pca_pcs, 0)
    data = pca_pcs / 30.0
    return data


def targets_to_cluster_ids(targets, cluster_names):
    d = {
        'Albanian': 'Albania',
        'Armenian': 'Armenia',
        'Belarusian': 'Belarus', 
        'Bulgarian': 'Bulgaria',
        'Cambodian': 'Cambodia',
        'Croatian': 'Croatia',
        'Czech': 'Czech Republic',
        'English': 'United Kingdom',
        'Estonian': 'Estonia',
        'Finnish': 'Finland',
        'French': 'France',
        'Georgian': 'Georgia',
        'Hungarian': 'Hungary',
        'Icelandic': 'Iceland',
        'Iranian': 'Iran',
        'Italian_North': 'Italy',
        'Italian_South': 'Italy',
        'Japanese': 'Japan',
        'Jordanian': 'Jordan',
        'Korean': 'South Korea',
        'Lebanese': 'Lebanon',
        'Lithuanian': 'Lithuania',
        'Norwegian': 'Norway',
        'Palestinian': 'Palestinian Territories',
        'Russian': 'Russia',
        'Scottish': 'United Kingdom',
        'Spanish': 'Spain',
        'Spanish_North': 'Spain',
        'Syrian': 'Syria',
        'Tajik': 'Tajikistan',
        'Thai': 'Thailand',
        'Turkish': 'Turkey',
        'Turkmen': 'Turkmenistan',
        'Ukrainian': 'Ukraine',
        'Uzbekistan': 'Uzbek',
    }

    ct = 0
    res = []
    for target in targets:
        if target in d:
            try:
                res.append(cluster_names.index(d[target]))
                ct += 1
            except: 
                res.append(len(cluster_names))
        else:
            res.append(len(cluster_names))
    print(f'{ct}/{len(targets)} converted to use cluster labels')
    return res


def get_modern_eurasia(batch_size, train=True, flattening=False, labels=False, wanted_labels=None, n_labels=-1, shuffle=None):
    if wanted_labels is not None or n_labels != -1 or (not flattening):
        raise NotImplementedError

    pcs = get_modern_eurasia_pcs()
    targets = get_modern_eurasia_labels(train)
    cluster_names = None
    if labels:
        if cluster_names is None:
            raise NotImplementedError
        cluster_ids = targets_to_cluster_ids(targets, cluster_names)
        dataset = TensorDataset(torch.tensor(pcs, dtype=torch.float), torch.tensor(cluster_ids, dtype=torch.long))
    else:
        dataset = TensorDataset(torch.tensor(pcs, dtype=torch.float))
        dataset = CustomDataset(dataset)

    if train:
        dataset = ConcatDataset([dataset] * 5)

    if shuffle is None:
        shuffle = train
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
    return dataloader


def get_modern_eurasia_labels(train):
    labels = _get_modern_eurasia_labels()
    if train:
        return np.asarray(list(labels) * 5)
    return labels