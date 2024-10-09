import math
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def scatter_plot_classes(Y, metadata, title=None, output_dir=None, show=True, categories=None):
    """
    creates a scatter plot visualization where each sample is colored by class
    :param Y: the 2D transformed data as numpy array
    :param metadata: a pandas Series or 1d numpy array containing metadata with classes
    :param title: optional title, if None the Series title is used
    :param output_dir: path to save the plot. If None plot won't be saved
    :param show: whether plt.show() is called
    :param categories: list of category/class names. If None they will be extracted from metadata
    """

    # Plot the PHATE transformed data
    if show:
        plt.figure(figsize=(24, 18))
        plt.rcParams['font.size'] = 16

    dot_size = 300 * (1 / len(metadata)**0.5)

    if categories is None:
        categories = metadata.unique()

    # Color coding
    if len(categories) > 40:
        colors = generate_pastel_colors(len(categories))
    else:
        colors = tab_forty()

    #plotting
    for i, category in enumerate(categories):
        mask = metadata == category
        color = colors[i]

        plt.scatter(Y[mask, 0], Y[mask, 1], color=color, label=category, s=dot_size)

    markerscale_legend = 50 / dot_size
    plt.legend(title=metadata.name, markerscale=markerscale_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    if title is None:
        title = metadata.name
    if show:
        plt.title(title, fontsize=22)
    else:
        plt.title(title)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if show:
        plt.show()
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        file_path = output_dir + title
        print("saving plot in " + file_path)
        plt.savefig(file_path, bbox_inches='tight')

def scatter_plot_gradient(Y, metadata, title=None, output_dir=None, show=True, gradient_range=None):
    """
    creates a scatter plot visualization where samples are colored by a color gradient
    :param Y: the output of PHATE's model.fit_transform(data)
    :param metadata: a pandas Series or 1d numpy array containing metadata with numbers
    :param title: optional title, if None the Series title is used
    :param save: path to save the plot. If None plot.show() will be called
    :param show: whether plt.show() is called
    :param gradient_range: gradient_range[0] is the minimum, gradient_range[1] is the maximum. They determine the range of the colorbar. If None they will be extracted from metadata
    """

    # Plot the PHATE transformed data
    if show:
        plt.figure(figsize=(24, 18))
        plt.rcParams['font.size'] = 16

    dot_size = 300 * (1 / len(metadata)**0.5)

    if gradient_range is None:
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=metadata, cmap='viridis', s=dot_size)
    else:
        min = gradient_range[0]
        max = gradient_range[1]
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=metadata, cmap='viridis', s=dot_size, vmin=min, vmax=max)
    plt.colorbar(scatter)

    if title is None:
        title = metadata.name
    if show:
        plt.title(title, fontsize=22)
    else:
        plt.title(title)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if show:
        plt.show()
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        file_path = output_dir + title
        print("saving plot in " + file_path)
        plt.savefig(file_path, bbox_inches='tight')

def pairwise_comparison(original, reconstructed, sample_names=None, output_file=None):
    """
    Plots multiple subplots, each containing two lines that represent gene expression samples. If N > 8, only the first 8 samples will be plotted
    :param original: NxD np array or pd DataFrame of original gene expression data
    :param reconstructed: NxD np array or pd DataFrame of reconstructed gene expression data
    :param title: A optional string that contains the plot title
    :param sample_names: a list of strings with the title of each sample. len(sample_names) should be equal to N
    :return: None
    """

    N, D = original.shape
    line_size = 1

    if N > 8:
        N = 8

    if sample_names is None:
        sample_names = [f"Sample {i + 1}" for i in range(N)]

    fig, axes = plt.subplots(N, 1, figsize=(40, 4 * N), sharex=True)

    if N == 1:
        axes = [axes]  # Make sure axes is iterable if there's only one subplot

    plt.rcParams.update({'font.size': 24})

    for i in range(N):
        axes[i].plot(original[i], label="ground truth", linewidth=line_size)
        axes[i].plot(reconstructed[i], label="prediction", linewidth=line_size)
        axes[i].legend(loc='upper right')
        axes[i].set_title(sample_names[i])

    plt.xlabel('Index')
    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
    plt.close()


def generate_pastel_colors(num_colors):
    # Diyuans color generator

    import random

    def get_random_color(pastel_factor=0.5):
        return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]

    def color_distance(c1, c2):
        return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])

    def generate_new_color(existing_colors, pastel_factor=0.5):
        max_distance = None
        best_color = None
        for i in range(0, 100):
            color = get_random_color(pastel_factor=pastel_factor)
            if not existing_colors:
                return color
            best_distance = min([color_distance(color, c) for c in existing_colors])
            if not max_distance or best_distance > max_distance:
                max_distance = best_distance
                best_color = color
        return best_color

    colors = []
    for i in range(0, num_colors):
        new_color = generate_new_color(colors, pastel_factor=0.9)
        colors.append(new_color)

    return colors

def tab_forty():
    # Generate distinct colors for each category
    colormap_one = plt.colormaps.get_cmap('tab20')
    colors_one = [colormap_one(i) for i in range(20)]
    colormap_two = plt.colormaps.get_cmap('tab20b')
    colors_two = [colormap_two(i) for i in range(20)]
    tab20_duplicated = colors_one + colors_two
    return tab20_duplicated


def plot_results(metrics, model, train_set, val_set, plot_dir):
    """
    plots the results and performance of the model
    :param metrics: a set containing all relevant metrics
    :param model: the trained model
    :param train_set: the training set
    :param val_set: the validation set
    :param plot_dir: path to save the plot
    :return:
    """
    current_epoch = len(metrics["train_loss"])
    plot_dir = plot_dir + f"/epoch {current_epoch}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    epochs = range(1, current_epoch + 1)

    plt.rcParams.update({'font.size': 16})

    large_loss_variance = (difference_greater_than(metrics["train_loss"], 8) or
                           difference_greater_than( metrics["val_loss"], 8))

    # Plot overall loss curves
    plt.figure(figsize=(16, 10))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if large_loss_variance:
        plt.yscale("log")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.savefig(plot_dir + "/loss_curves.png")
    plt.close()

    # Plot R2 score curves
    plt.figure(figsize=(16, 10))
    plt.plot(epochs, metrics["train_R2"], label="Train R2")
    plt.plot(epochs, metrics["val_R2"], label="Validation R2")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.title("Train and Validation R2 Score")
    plt.legend()
    plt.savefig(plot_dir + "/r2_curves.png")
    plt.close()

    # Plot all train and validation losses
    plt.figure(figsize=(16, 10))
    plt.plot(epochs, metrics["train_recon"], label="Train Reconstruction Loss", linestyle="--", color="royalblue")
    plt.plot(epochs, metrics["train_reg"], label="Train Purity Loss", linestyle="--", color="orange")
    plt.plot(epochs, metrics["train_kl"], label="Train KLD Loss", linestyle="--", color="saddlebrown")
    plt.plot(epochs, metrics["val_recon"], label="Validation Reconstruction Loss", linestyle=":", color="royalblue")
    plt.plot(epochs, metrics["val_reg"], label="Validation Purity Loss", linestyle=":", color="orange")
    plt.plot(epochs, metrics["val_kl"], label="Validation KLD Loss", linestyle=":", color="brown")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if large_loss_variance:
        plt.yscale("log")
    plt.title("Multitask Train Losses")
    plt.legend()
    plt.savefig(plot_dir + "/all_losses.png")
    plt.close()

    # Plot the two weights
    plt.figure(figsize=(16, 10))
    plt.plot(epochs, metrics["w1"], label="Reconstruction Weight")
    plt.plot(epochs, metrics["w2"], label="Purity Weight")
    plt.xlabel("Epochs")
    plt.ylabel("Weight")
    plt.title("Task Weights")
    plt.legend()
    plt.savefig(plot_dir + "/task_weights.png")
    plt.close()

    # Plot the reconstructed samples and predicted purities against the original values
    viz_train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    viz_val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)
    visualize_model_outputs(model, plot_dir, viz_train_loader, dataset="train")
    visualize_model_outputs(model, plot_dir, viz_val_loader, dataset="validation")


def visualize_model_outputs(model, plot_dir, data_loader, dataset=""):
    device_used = get_model_device(model)
    model.cpu().eval()
    with torch.no_grad():
        x, w, _ = next(iter(data_loader))
        x_hat, w_hat, _, _ = model(x)
    if dataset != "":
        dataset = dataset + "_"
    # plot reconstruction against original samples
    pairwise_comparison(x, x_hat, output_file=plot_dir + f"/{dataset}samples_gene_expression.png")
    # plot predicted against original purity
    w = w.view(1, -1)
    w_hat = w_hat.view(1, -1)
    pairwise_comparison(w, w_hat, output_file=plot_dir + f"/{dataset}samples_purity.png")
    model.to(device_used)


def get_model_device(model):
    return next(model.parameters()).device

def difference_greater_than(x, max_difference):
    max = np.array(x).max()
    min = np.array(x).min()
    difference = max - min
    return difference > max_difference



def tumor_percentage_histogram(data, dataset_name, bins=20, dataset=None):
    """
    Plots a histogram of the tumor percentage of samples, with optional coloring based on dataset (e.g., 'TCGA', 'CCLE').

    :param data: a Series containing the tumor percentage
    :param dataset_name: the name for the plot, e.g., "TCGA" or "BRCA"
    :param bins: number of bins for the histogram
    :param dataset: a Series (optional) containing the dataset information (e.g., 'TCGA' or 'CCLE') corresponding to 'data'
    :return: None
    """
    plt.xlim(0., 1.)
    plt.xlabel("Tumor Percentage")
    plt.ylabel("Number of samples")
    plt.title(f"{dataset_name} Tumor Percentage")

    if dataset is not None:
        # Ensure all dataset names are in uppercase
        dataset = dataset.str.upper()

        # Sort the dataset so 'TCGA' is first and 'CCLE' is second, if they exist
        ordered_datasets = ['TCGA', 'CCLE']
        unique_datasets = [ds for ds in ordered_datasets if ds in dataset.unique()]
        colors = ['#1f77b4', '#ff7f0e']  # Define colors for TCGA and CCLE
        hist_data = []

        # Calculate histogram data for each dataset
        for i, ds in enumerate(unique_datasets):
            subset_data = data[dataset == ds]
            hist, bin_edges = np.histogram(subset_data, bins=bins, range=(0., 1.))
            hist_data.append(hist)

            # Calculate the width of each bar
            bin_width = bin_edges[1] - bin_edges[0]

            # Plot the current dataset with corrected alignment
            if i == 0:
                plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', color=colors[i], label=f"{ds}")
            else:
                plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', color=colors[i], bottom=hist_data[0], label=f"{ds}")

        plt.legend()
    else:
        # Default behavior with no dataset provided
        plt.hist(data, bins=bins, range=(0., 1.))

    plt.show()

def metrics_overview_plot(df, label=None, xlabel=None):
    """
    creates a overview plot of metrics & losses
    :param df: a DataFrame. Every column contains one metric and the indices must be numbers
    :param label: if not None, it will be used as the label for the plots legend
    :return: None
    """
    num_metrics = len(df.columns)
    axis_length = math.ceil(math.sqrt(num_metrics))
    indices = df.index

    for i in range(num_metrics):
        plt.subplot(axis_length, axis_length, i + 1)
        column_name = df.columns[i]
        plt.title(column_name)
        plt.plot(indices, df[column_name], label=label)

        if label is not None:
            plt.legend()

        if xlabel is not None:
            plt.xlabel(xlabel)

    plt.tight_layout()


def plot_phate_per_disease(diseases_dict, column_name, as_classes):
    """
    generates a large plot containing subplots with PHATE visualisation for each disease with at least two samples
    :param diseases_dict: the dictionary created by
    :param column_name: the name of the column to use for coloring the dots
    :param as_classes: if True the data in column name will be plotted as colored classes, otherwise as color gradient
    :return: None
    """
    number_diseases = len(diseases_dict)

    axis_length = math.ceil(math.sqrt(number_diseases))

    plt.figure(figsize=(10 * axis_length, 10 * axis_length))

    if as_classes:
        # extract all possible categories from the dictionary. Ensures that in the final plot all subplots will have
        # the same labels -> same order and same color for all classes
        combined_meta = combine_per_disease_dict(diseases_dict)
        diseases = combined_meta[column_name].unique()
    else:
        combined_meta = combine_per_disease_dict(diseases_dict)
        min = combined_meta[column_name].min()
        max = combined_meta[column_name].max()
        gradient_range = (min, max)

    for i, cancer in enumerate(diseases_dict):
        disease_Y, disease_meta = diseases_dict[cancer]
        plt.subplot(axis_length, axis_length, i + 1)
        if as_classes:
            scatter_plot_classes(disease_Y, disease_meta[column_name], title=cancer, output_dir=None, show=False, categories=diseases)
        else:
            scatter_plot_gradient(disease_Y, disease_meta[column_name], title=cancer, output_dir=None, show=False, gradient_range=gradient_range)

    plt.suptitle(column_name, fontsize="xx-large", y=0.05)
    plt.tight_layout()
    plt.show()
    plt.close()


def combine_per_disease_dict(diseases_dict):
    meta_list = []
    for disease_name, data in diseases_dict.items():
        genes, meta = data
        meta_list.append(meta)
    combined_meta = pd.concat(meta_list)
    return combined_meta
