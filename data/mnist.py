import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms as tsfm
from torch.utils.data import DataLoader, ConcatDataset

from .utils import CustomDataset


def get_mnist(batch_size, train=True, flattening=False, labels=False, wanted_labels=None, n_labels=-1, shuffle=None, toy=False, seed=None):
    # load dataset from the hub
    dataset = MNIST('.', 
                    train=train, 
                    download=True, 
                    transform=tsfm.ToTensor())
    dataset = CustomDataset(dataset, flattening=flattening, use_label=labels, wanted_labels=wanted_labels, n_labels=n_labels, toy=toy, seed=seed)

    if toy:
        dataset = ConcatDataset([dataset] * 5000)

    if shuffle is None:
        shuffle = train
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
    return dataloader


def get_mnist_labels(train, toy=False):
    dataset = MNIST('.', 
                    train=train, 
                    download=True)
    if not toy:
        return dataset.targets.numpy()
    else:
        return np.asarray(list(dataset.targets.numpy()[:4]) * 5000)