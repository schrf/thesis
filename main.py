import pickle

import pandas as pd
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_visualization import pairwise_comparison
from src.training import epochs_loop
from src.data_transformation import combine_ccle_tcga
from src.datasets.simple_dataset import SimpleDataset
import os
import datetime
from src.data_visualization import pairwise_comparison

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

    metrics, model = epochs_loop(train_loader, val_loader)

    plot_results(metrics, model, val_set, plot_comment=comment)


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

def plot_results(metrics, model, val_set, plot_comment=""):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m-%d-%H-%M")
    plot_dir = "plots/multitask/" + formatted_time + plot_comment
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Plot overall loss curves
    plt.figure()
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.savefig(plot_dir + "/loss_curves.png")

    # Plot R2 score curves
    plt.figure()
    plt.plot(epochs, metrics["train_R2"], label="Train R2")
    plt.plot(epochs, metrics["val_R2"], label="Validation R2")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.title("Train and Validation R2 Score")
    plt.legend()
    plt.savefig(plot_dir + "/r2_curves.png")

    # Plot all three train losses
    plt.figure()
    plt.plot(epochs, metrics["train_recon"], label="Train Reconstruction Loss")
    plt.plot(epochs, metrics["train_reg"], label="Train Purity Loss")
    plt.plot(epochs, metrics["train_kl"], label="Train KLD Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Multitask Train Losses")
    plt.legend()
    plt.savefig(plot_dir + "/train_losses.png")

    # Plot all three validation losses
    plt.figure()
    plt.plot(epochs, metrics["val_recon"], label="Validation Reconstruction Loss")
    plt.plot(epochs, metrics["val_reg"], label="Validation Purity Loss")
    plt.plot(epochs, metrics["val_kl"], label="Validation KLD Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Multitask Validation Losses")
    plt.legend()
    plt.savefig(plot_dir + "/val_losses.png")

    # Plot the reconstructed samples and predicted purities against the original values
    viz_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)
    model = model.cpu().eval()

    with torch.no_grad():
        x, w, _ = next(iter(viz_loader))
        x_hat, w_hat, _, _ = model(x)

    # plot reconstruction against original samples
    pairwise_comparison(x, x_hat, output_file=plot_dir + "/samples_gene_expression.png")

    # plot predicted against original purity
    w = w.view(1, -1)
    w_hat = w_hat.view(1, -1)
    pairwise_comparison(w, w_hat, output_file=plot_dir + "/samples_purity.png")


if __name__ == "__main__":
    main()
