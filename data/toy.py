import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

from .utils import CustomDataset


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]


def get_toy_data(name, batch_size, train=True):
    with open(f'data/{name}.npy', 'rb') as f:
        data = np.load(f)
    dataset = TensorDataset(torch.Tensor(data))
    dataset = CustomDataset(dataset)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return dataloader