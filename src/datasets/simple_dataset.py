import torch
import pandas as pd
from src.datasets.base_dataset import Dataset


class SimpleDataset(Dataset):
    def __init__(self, genes: pd.DataFrame, meta: pd.DataFrame, transform=None):
        super().__init__(genes, meta, transform)

    def __getitem__(self, index):
        return (torch.tensor(self.genes[index], dtype=torch.float32),
                torch.tensor([self.meta["cancer_purity"].iloc[index]], dtype=torch.float32),
                self.indices[index])