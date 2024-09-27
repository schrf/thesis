import pandas as pd
import torch
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


def splitted_metrics(model, filter, genes, meta):
    """
    calculates the metrics and losses for each the filtered data and the rest
    :param model: the trained model for evaluation
    :param filter: the filter to apply
    :param genes: a DataFrame containing gene expression values
    :param meta: a DataFrame containing the coresponding metadata
    :return: filter_metrics and other_metrics, each containing total_loss,
    total_recon_loss, total_reg_loss,  total_kl_loss, total_r2
    """
    inverse_filter = ~filter

    filter_genes = genes[filter]
    other_genes = genes[inverse_filter]

    filter_meta = meta[filter]
    other_meta = meta[inverse_filter]

    transform = {
        "z_score": "per_sample"
    }

    filter_dataset = SimpleDataset(filter_genes, filter_meta, transform=transform)
    other_dataset = SimpleDataset(other_genes, other_meta, transform=transform)
    max_batch_size = 1
    filter_dataloader = create_dataloader(filter_dataset, max_batch_size)
    other_dataloader = create_dataloader(other_dataset, max_batch_size)
    filter_metrics = val_loop(model, filter_dataloader, [0.85, 0.15], 0.00001, "cuda")
    other_metrics = val_loop(model, other_dataloader, [0.85, 0.15], 0.00001, "cuda")
    return filter_metrics, other_metrics


def splitted_dataframes(model_paths, number_mixed_list, filter, genes, meta):
    """
    creates two DataFrames for filter and rest data containing the metrics and losses for all trained models
    :param model_paths: a list of file paths to the models
    :param number_mixed_list: a list of the number of mixed samples. Must be the same length as models
    :param filter: the filter to apply
    :param genes: a DataFrame containing gene expression values
    :param meta: a DataFrame containing the corresponding metadata
    :return: two DataFrames with columns for each
    """
    filter_dict = {
        "total_recon_loss": [],
        "total_reg_loss": [],
        "total_r2": []
    }
    other_dict = {
        "total_recon_loss": [],
        "total_reg_loss": [],
        "total_r2": []
    }

    models = model_loader(model_paths)

    for model in models:
        filter_metrics, other_metrics = splitted_metrics(model, filter, genes, meta)
        extract_relevant_metrics(filter_metrics, filter_dict)
        extract_relevant_metrics(other_metrics, other_dict)

    filter_df = pd.DataFrame(filter_dict, index=number_mixed_list)
    other_df = pd.DataFrame(other_dict, index=number_mixed_list)

    return filter_df, other_df

def get_latent_representation(genes, model, device, transform={"z_score": "per_sample"}):
    """
    takes the gene expression data and passes it through the model to get the latent representation.
    Assumes there is enough gpu memory to pass everything at a time through the model
    :param genes: the gene expression dataframe
    :param model: the trained model object to use. Has to have a model.encoder.fc layer
    :param device: the device to be used
    :return:  returns the latent space representation of all samples as pd.Dataframe
    """
    indices = genes.index

    # add dummy cancer purity for compatibility with the SimpleDataset
    dummy_meta = pd.DataFrame({"cancer_purity": pd.Series(-1, index=indices)})

    dataset = SimpleDataset(genes, dummy_meta, transform)

    genes_tensor, _, _ = dataset[:]
    genes_tensor = genes_tensor.to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        print(model.encoder.resnet.fc)
        mu, sigma = model.encoder.forward(genes_tensor)
        # latent_representation = torch.cat((mu, sigma), dim=-1)
        latent_representation = sigma

    latent_df = pd.DataFrame(latent_representation.cpu().numpy())
    latent_df.index = indices

    return latent_df
