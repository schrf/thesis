import pandas as pd
import torch

from src.datasets.base_dataset import Dataset


class NoisyDataset(Dataset):
    """
    a Dataset that adds gaussian noise with a standard deviation of std to each sample in the dataset
    """
    def __init__(self, genes: pd.DataFrame, meta: pd.DataFrame, var, transform=None):
        super().__init__(genes, meta, transform)
        self.var = var

    def __getitem__(self, index):
        gene_exp = torch.tensor(self.genes[index], dtype=torch.float32)
        purity = torch.tensor([self.meta["cancer_purity"].iloc[index]], dtype=torch.float32)
        return self.add_gaussian_noise(gene_exp), purity, self.indices[index]

    def add_gaussian_noise(self, data):
        noise = (self.var**0.5)*torch.randn(data.shape)
        return data + noise