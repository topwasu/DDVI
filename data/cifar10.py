import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms as tsfm
from torch.utils.data import DataLoader

from .utils import CustomDataset


def get_cifar10(batch_size, train=True, flattening=False, labels=False, wanted_labels=None, n_labels=-1, shuffle=None, seed=None):
    transform = tsfm.Compose([
        tsfm.RandomHorizontalFlip(),
        tsfm.ToTensor(),
        tsfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # load dataset from the hub
    dataset = CIFAR10('.', 
                      train=train, 
                      download=True, 
                      transform=transform)
    dataset = CustomDataset(dataset, flattening=flattening, use_label=labels, wanted_labels=wanted_labels, n_labels=n_labels, seed=seed)

    if shuffle is None:
        shuffle = train
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
    return dataloader


def get_cifar10_labels(train):
    dataset = CIFAR10('.', 
                      train=train, 
                      download=True)
    return np.asarray(dataset.targets)