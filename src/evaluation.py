import pandas as pd
from torch.utils.data import DataLoader

from src.datasets.simple_dataset import SimpleDataset
from src.loader import model_loader
from src.training import val_loop


def extract_relevant_metrics(metrics_tuple, dict):
    """extracts relevant metrics for plotting and saves them into a dictionary"""
    _, low_recon, low_reg, _, low_r2 = metrics_tuple
    dict["total_recon_loss"].append(low_recon), dict["total_reg_loss"].append(low_reg), dict["total_r2"].append(low_r2)



def create_dataloader(dataset, max_batch_size):
    """creates a dataloader of max_batch_size or smaller if there are not enough samples"""
    number_samples = len(dataset)

    if number_samples > max_batch_size:
        dataloader = DataLoader(dataset, batch_size=max_batch_size, shuffle=False, drop_last=False)
    else:
        dataloader = DataLoader(dataset, batch_size=number_samples, shuffle=False, drop_last=False)

    return dataloader


def high_low_metrics(model, genes, meta):
    """
    calculates the metrics and losses for given gene expression values, metadata and a model
    :param model: the trained model for evaluation
    :param genes: a DataFrame containing gene expression values
    :param meta: a DataFrame containing the coresponding metadata
    :return: low_metrics and high_metrics, each containing total_loss,
    total_recon_loss, total_reg_loss,  total_kl_loss, total_r2
    """
    low_filter = meta["cancer_purity"] < 0.6
    high_filter = meta["cancer_purity"] >= 0.6

    low_genes = genes[low_filter]
    high_genes = genes[high_filter]

    low_meta = meta[low_filter]
    high_meta = meta[high_filter]

    transform = {
        "z_score": "per_sample"
    }

    low_dataset = SimpleDataset(low_genes, low_meta, transform=transform)
    high_dataset = SimpleDataset(high_genes, high_meta, transform=transform)
    max_batch_size = 1024
    low_dataloader = create_dataloader(low_dataset, max_batch_size)
    high_dataloader = create_dataloader(high_dataset, max_batch_size)
    low_metrics = val_loop(model, low_dataloader, [0.85, 0.15], 0.00001, "cuda")
    high_metrics = val_loop(model, high_dataloader, [0.85, 0.15], 0.00001, "cuda")
    return low_metrics, high_metrics


def high_low_purity_dataframes(model_paths, number_mixed_list, genes, meta):
    """
    creates two DataFrames for low and high purity containing the metrics and losses for all trained models
    :param model_paths: a list of file paths to the models
    :param number_mixed_list: a list of the number of mixed samples. Must be the same length as models
    :param genes: a DataFrame containing gene expression values
    :param meta: a DataFrame containing the coresponding metadata
    :return: two DataFrames with columns for each
    """
    low_dict = {
        "total_recon_loss": [],
        "total_reg_loss": [],
        "total_r2": []
    }
    high_dict = {
        "total_recon_loss": [],
        "total_reg_loss": [],
        "total_r2": []
    }

    models = model_loader(model_paths)

    for model in models:
        low_metrics, high_metrics = high_low_metrics(model, genes, meta)
        extract_relevant_metrics(low_metrics, low_dict)
        extract_relevant_metrics(high_metrics, high_dict)

    low_df = pd.DataFrame(low_dict, index=number_mixed_list)
    high_df = pd.DataFrame(high_dict, index=number_mixed_list)

    return low_df, high_df
