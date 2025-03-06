import logging
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from cyvcf2 import VCF
from torchvision.datasets import MNIST
from torchvision import transforms as tsfm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from .utils import CustomDataset


log = logging.getLogger(__name__)


def _get_1kgenome_labels():
    # Reference: https://github.com/diazale/1KGP_dimred/blob/master/Genotype_dimred_demo.ipynb

    vcf_name = '/share/kuleshov/pop_gen_data/1kgenome/ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz'
    pop_desc_name = '/share/kuleshov/pop_gen_data/1kgenome/20131219.populations.tsv'
    pop_file_name = '/share/kuleshov/pop_gen_data/1kgenome/affy_samples.20141118.panel'

    # get samples
    individuals = VCF(vcf_name).samples

    # get pop by continent
    name_by_code = {}
    pop_by_continent = defaultdict(list)
    with open(pop_desc_name ,'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split('\t')
        if split_line[0] in ['Population Description','Total','']:  # header or footer
            continue
        name_by_code[split_line[1]] = split_line[0]
        pop_by_continent[split_line[2]].append(split_line[1])
    continents = list(pop_by_continent.keys())
    pops=[]
    for continent in continents:
        pops.extend(pop_by_continent[continent])

    # get pop by individ and individs by pop
    population_by_individual = defaultdict(int)
    individuals_by_population = defaultdict(list)
    with open(pop_file_name ,'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split()
        if split_line[0] == 'sample':  # header line
            continue
        sample_name = split_line[0]
        population_name = split_line[1]
        population_by_individual[sample_name] = population_name
        individuals_by_population[population_name].append(sample_name) 

    # # get indices of pop members by pop
    # indices_of_population_members = defaultdict(list)
    # for idx, individual in enumerate(individuals):
    #     try:
    #         indices_of_population_members[population_by_individual[individual]].append(idx)
    #     except KeyError: # We do not have population info for this individual
    #         continue
    # return 
    targets = []
    for ind in individuals:
        pop = population_by_individual[ind]
        targets.append((pop, name_by_code[pop]))
    numbered_targets = np.asarray([pops.index(target[0]) for target in targets])
    return numbered_targets


def get_1kgenome_pcs():
    with open('/share/kuleshov/pop_gen_data/1kgenome/new_1kgenome_1000pcs.npy', 'rb') as f:
        pca_pcs = np.load(f)
    # pca_pcs = pca_pcs[:, :1000]
    # data = (pca_pcs - np.mean(pca_pcs, 0)) / np.std(pca_pcs, 0)
    data = pca_pcs / 30.0
    return data


def get_1kgenome_small_pcs():
    with open('/share/kuleshov/pop_gen_data/1kgenome/new_1kgenome_15pcs.npy', 'rb') as f:
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


def get_1kgenome(batch_size, train=True, flattening=False, labels=False, wanted_labels=None, n_labels=-1, shuffle=None, small_pcs=False):
    if wanted_labels is not None or n_labels != -1 or (not flattening):
        raise NotImplementedError

    if small_pcs:
        pcs = get_1kgenome_small_pcs()
    else:
        pcs = get_1kgenome_pcs()
    targets = get_1kgenome_labels(train)
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


def get_1kgenome_labels(train):
    labels = _get_1kgenome_labels()
    if train:
        return np.asarray(list(labels) * 5)
    return labels