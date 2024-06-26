import pandas as pd
import torch

from src.datasets.base_dataset import Dataset


class NoisyDataset(Dataset):
    """
    a Dataset that adds gaussian noise with a standard deviation of std to each sample in the dataset
    """
    def __init__(self, data: pd.DataFrame, var, transform=None):
        super().__init__(data, transform)
        self.var = var

    def __getitem__(self, index):
        tensor = torch.tensor(self.data[index], dtype=torch.float32)
        return self.add_gaussian_noise(tensor), self.indices[index]

    def add_gaussian_noise(self, data):
        noise = (self.var**0.5)*torch.randn(data.shape)
        return data + noise