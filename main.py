import pandas as pd
import sys
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from src.training import num_epochs, input_dim, device, epochs_loop
from src.data_transformation import filter_variance
from src.datasets.simple_dataset import SimpleDataset
import os

# hyperparameter initialization

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 3:
        print("Usage: python script.py <ccle_path> <tcga_path>")
        sys.exit(1)

    # train-val split should stay the same using a seed
    torch.manual_seed(42)

    ccle_path = sys.argv[1]
    tcga_path = sys.argv[2]

    try:
        data = load_data(ccle_path, tcga_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"An error occurred during loading ccle and tcga: {e}")
        sys.exit(1)

    train_set, val_set = train_val_split(data)

    transform = {
        "genes_filter": filter_variance(data, input_dim),
        "z_score": "per_sample"
    }

    train_set = SimpleDataset(train_set, transform=transform)
    val_set = SimpleDataset(val_set, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

    print(f"{device} is used")

    metrics = epochs_loop(train_loader, val_loader)

    plot_metrics(metrics)

def load_data(ccle_path, tcga_path):
    ccle = pd.read_csv(ccle_path, index_col=0)
    ccle = ccle.fillna(0)
    tcga = pd.read_csv(tcga_path, index_col=0)
    tcga = tcga.fillna(0)
    combined = pd.concat([ccle, tcga], join="inner")
    return combined

def train_val_split(data):
    train_size = int(0.8 * len(data))
    train_data = data.sample(train_size)
    val_data = data.drop(train_data.index)
    return train_data, val_data


def plot_metrics(metrics):
    plot_dir = "logs/variational autoencoder"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    epochs = range(1, num_epochs + 1)

    # Plot loss curves
    plt.figure()
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.savefig("plots/simple/loss_curves.png")

    # Plot R2 score curves
    plt.figure()
    plt.plot(epochs, metrics["train_R2"], label="Train R2")
    plt.plot(epochs, metrics["val_R2"], label="Validation R2")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.title("Train and Validation R2 Score")
    plt.legend()
    plt.savefig("plots/simple/r2_curves.png")

if __name__ == "__main__":
    main()
