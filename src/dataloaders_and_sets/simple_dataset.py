import torch
import pandas as pd
from src.dataloaders_and_sets.base_dataset import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        super().__init__(data, transform)

    def __getitem__(self, index):
        return torch.tensor(self.data.iloc[index], dtype=torch.float32)