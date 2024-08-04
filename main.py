import pickle

import sys
import torch
from sklearn.model_selection import train_test_split

from src.data_visualization import plot_results
from src.training import epochs_loop
from src.data_transformation import combine_ccle_tcga
from src.datasets.simple_dataset import SimpleDataset


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python main.py <ccle_path> <tcga_path> [optional comment for run]")
        sys.exit(1)

    ccle_path = sys.argv[1]
    tcga_path = sys.argv[2]
    if len(sys.argv) == 4:
        comment = sys.argv[3]
    else:
        comment = ""

    try:
        genes, meta = load_data(ccle_path, tcga_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"An error occurred during loading ccle and tcga: {e}")
        sys.exit(1)

    train_genes, val_genes, train_meta, val_meta = train_test_split(genes, meta, test_size=0.2, random_state=42)

    transform = {
        "z_score": "per_sample"
    }

    train_set = SimpleDataset(train_genes, train_meta, transform=transform)
    val_set = SimpleDataset(val_genes, val_meta, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

    metrics, model = epochs_loop(train_loader, val_loader, val_set, comment)

def load_data(ccle_path, tcga_path):
    """
    returns the combined ccle and tcga datasets
    :param ccle_path: path to ccle pickle file
    :param tcga_path: path to tcga pickle file
    :return: gene expression and metadata dataframe
    """

    # Open the pickle file in binary read mode
    with open(ccle_path, 'rb') as file:
        # Load the contents of the pickle file
        ccle = pickle.load(file)

    with open(tcga_path, 'rb') as file:
        tcga = pickle.load(file)

    ccle_genes = ccle["rnaseq"]
    ccle_meta = ccle["meta"]
    tcga_genes = tcga["rnaseq"]
    tcga_meta = tcga["meta"]
    del ccle, tcga

    combined_genes, combined_meta = combine_ccle_tcga(ccle_genes, ccle_meta, tcga_genes, tcga_meta)
    return combined_genes, combined_meta


if __name__ == "__main__":
    main()
