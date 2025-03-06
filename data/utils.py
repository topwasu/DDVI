import numpy as np
import torch
from torch.utils.data import Dataset

import logging
log = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, dataset, flattening=False, use_label=False, wanted_labels=None, n_labels=-1, toy=False, seed=None):
        self.dataset = dataset
        self.flattening = flattening
        self.use_label = use_label

        if self.use_label:
            self.targets = np.asarray(dataset.targets)
            if n_labels != -1:
                if seed is not None:
                    np.random.seed(seed)
                self.targets[np.random.choice(50000, 50000 - n_labels, replace=False)] = 10

        if wanted_labels is not None:
            indices = []
            for lab in wanted_labels:
                indices.append(np.nonzero(self.dataset.targets == lab))
            self.indices = np.concatenate(indices, axis=None)
            log.info(self.indices)
        else:
            if not toy:
                self.indices = np.arange(len(self.dataset))
            else:
                self.indices = np.arange(4)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        if self.flattening:
            x = torch.flatten(self.dataset[idx][0], start_dim=0)
        else:
            x = self.dataset[idx][0]

        if self.use_label:
            return x, self.targets[idx]
        else:
            return x
