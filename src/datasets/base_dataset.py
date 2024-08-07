import sys
import os
from typing import Tuple

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import z_score normalization functions
from src.data_transformation import z_score_normalization_rowwise, z_score_normalization_columnwise, filter_variance

from abc import ABC, abstractmethod
import pandas as pd

class Dataset(ABC):
    """
    Abstract base class for all datasets
    """
    def __init__(self, genes: pd.DataFrame, meta: pd.DataFrame, transform=None):
        """
        initializes the dataset
        :param data: data (without any additional labels)
        :param transform: a set that contains the transformation applied to the data
        """
        self.transform = transform
        if self.transform is not None:
            if self.transform.get("genes_filter") is not None:
                genes_list = self.transform.get("genes_filter")
                genes = genes[genes_list]

            if self.transform.get("z_score") == "per_gene":
                genes = z_score_normalization_columnwise(genes, genes.columns)
            elif self.transform.get("z_score") == "per_sample":
                genes = z_score_normalization_rowwise(genes, genes.columns)
        self.indices = genes.index
        self.genes = genes.to_numpy()
        self.meta = meta

    @abstractmethod
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return 1. data sample at given index and transform it according to the transform set; 2. Returns the index"""

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.genes)

