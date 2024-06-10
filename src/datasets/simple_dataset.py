import torch
import pandas as pd
from src.datasets.base_dataset import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        super().__init__(data, transform)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)