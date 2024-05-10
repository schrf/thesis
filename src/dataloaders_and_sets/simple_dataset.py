import torch

from src.dataloaders_and_sets.base_dataset import Dataset


class SimpleDataset(Dataset):
    def __getitem__(self, index):
        return torch.tensor(self.data.iloc[index], dtype=torch.float32)