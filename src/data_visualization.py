import math
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch


def scatter_plot_classes(Y, metadata, title=None, output_dir=None):
    """
    creates a scatter plot visualization where each sample is colored by class
    :param Y: the 2D transformed data as numpy array
    :param metadata: a pandas Series or 1d numpy array containing metadata with classes
    :param title: optional title, if None the Series title is used
    :param output_dir: path to save the plot. If None plot.show() will be called
    """

    # Plot the PHATE transformed data
    plt.figure(figsize=(24, 18))
    plt.rcParams['font.size'] = 16

    dot_size = 300 * (1 / len(metadata)**0.5)

    # Color coding
    categories = metadata.unique()

    if len(categories) > 40:
        colors = generate_pastel_colors(len(categories))
    else:
        colors = tab_forty()

    #plotting
    for i, category in enumerate(categories):
        mask = metadata == category
        color = colors[i]

        plt.scatter(Y[mask, 0], Y[mask, 1], color=color, label=category, s=dot_size)

    plt.legend(title=metadata.name, markerscale=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    if title is None:
        title = metadata.name
    plt.title(title, fontsize=22)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if output_dir is None:
        plt.show()
    else:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        file_path = output_dir + title
        print("saving plot in " + file_path)
        plt.savefig(file_path, bbox_inches='tight')

def scatter_plot_gradient(Y, metadata, title=None, output_dir=None):
    """
    creates a scatter plot visualization where samples are colored by a color gradient
    :param Y: the output of PHATE's model.fit_transform(data)
    :param metadata: a pandas Series or 1d numpy array containing metadata with numbers
    :param title: optional title, if None the Series title is used
    :param save: path to save the plot. If None plot.show() will be called
    """

    # Plot the PHATE transformed data
    plt.figure(figsize=(24, 18))
    plt.rcParams['font.size'] = 16

    dot_size = 300 * (1 / len(metadata)**0.5)

    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=metadata, cmap='viridis', s=dot_size)
    plt.colorbar(scatter)

    if title is None:
        title = metadata.name
    plt.title(title, fontsize=22)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if output_dir is None:
        plt.show()
    else:
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

def tumor_percentage_histogram(data, dataset_name, bins=20):
    """
    plots a histogram of the tumor percentage of samples
    :param data: a Series containing the tumor percentage
    :param dataset_name: the name, e.g. "TCGA" or "BRCA"
    :return: None
    """
    plt.xlim(0., 1.)
    plt.hist(data, bins=bins, range=(0., 1.))
    plt.xlabel("Tumor Percentage")
    plt.ylabel("Number of samples")
    plt.title(f"{dataset_name} Tumor Percentage")
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