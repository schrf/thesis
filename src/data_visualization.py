import os.path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import phate

def scatter_plot_classes(Y, metadata, title=None, output_dir=None):
    """
    creates a scatter plot visualization where each sample is colored by class
    :param data: the gene expression data as pandas dataframe or 2d numpy array
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
        axes[i].plot(original[i], label="original gene expression data", linewidth=line_size)
        axes[i].plot(reconstructed[i], label="reconstructed gene expression data", linewidth=line_size)
        axes[i].legend(loc='upper right')
        axes[i].set_title(sample_names[i])

    plt.xlabel('Index')
    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)


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