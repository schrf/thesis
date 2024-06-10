import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import z_score normalization functions
from src.data_transformation import z_score_normalization_rowwise, z_score_normalization_columnwise, filter_variance

from abc import ABC, abstractmethod
import pandas as pd

class Dataset(ABC):
    """
    Abstract base class for all datasets
    """
    def __init__(self, data: pd.DataFrame, transform=None):
        """

        :param data: data (without any additional labels)
        :param transform: a set that contains the transformation applied to the data
        """
        self.transform = transform
        if self.transform is not None:
            if self.transform.get("genes_filter") is not None:
                genes_list = self.transform.get("genes_filter")
                data = data[genes_list]

            if self.transform.get("z_score") == "per_gene":
                data = z_score_normalization_columnwise(data, data.columns)
            elif self.transform.get("z_score") == "per_sample":
                data = z_score_normalization_rowwise(data, data.columns)

        self.data = data.to_numpy()

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index and transform it according to the transform set"""

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

