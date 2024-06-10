from pacmap import PaCMAP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.data_transformation import z_score_normalization_rowwise, z_score_normalization_columnwise, filter_variance


def combined_data_pacmap(data, viz_preprocessing):
    # Extract the data columns for Pacmap visualization
    data_columns = [col for col in data.columns if col not in ['primary_disease', 'gender', 'age', 'dataset', 'normalized_age']]

    if viz_preprocessing["z_score_norm"] == "per_gene":
        data = z_score_normalization_columnwise(data, data_columns)
    elif viz_preprocessing["z_score_norm"] == "per_sample":
        data = z_score_normalization_rowwise(data, data_columns)

    _combined_data_pacmap(data, viz_preprocessing, data_columns)


def _combined_data_pacmap(data, viz_preprocessing, data_columns):
    # Step 1: Drop NaN values and normalize the age column
    data.dropna(inplace=True)
    data['normalized_age'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())

    # Initialize Pacmap
    model = PaCMAP()

    if viz_preprocessing["only_most_variant"] is not None:

        selected_columns = filter_variance(data[data_columns], viz_preprocessing["only_most_variant"])
    else:
        selected_columns = data_columns
    
    # Fit the model
    Y = model.fit_transform(data[selected_columns])

    # Visualizations
    create_pacmap_visualization(data, Y, 'primary_disease', 'Pacmap Visualization 5000 most variant genes: Primary Disease')
    create_pacmap_visualization(data, Y, 'gender', 'Pacmap Visualization 5000 most variant genes: Gender')
    create_pacmap_visualization(data, Y, 'age', 'Pacmap Visualization 5000 most variant genes: Age')
    create_pacmap_visualization(data, Y, 'dataset', 'Pacmap Visualization 5000 most variant genes: Dataset')

def create_pacmap_visualization(data, Y, metadata_column, title):
    """Helper function to create Pacmap visualization"""
    
    # Plot the Pacmap visualization
    plt.figure(figsize=(8, 6))

    dot_size = 1
    
    # Color coding
    if metadata_column == 'age':
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=data['normalized_age'], cmap='viridis', s=dot_size)
        plt.colorbar(scatter, label='Relative Age')
    else:
        categories = data[metadata_column].unique()
        num_categories = len(categories)
        
        # Generate distinct colors for each category
        colormap_one = plt.colormaps.get_cmap('tab20')
        colors_one = [colormap_one(i) for i in range(20)]
        colormap_two = plt.colormaps.get_cmap('tab20b')
        colors_two = [colormap_two(i) for i in range(20)]
        tab20_duplicated = colors_one + colors_two
        colors = generate_pastel_colors(num_categories)
                
        for i, category in enumerate(categories):
            mask = data[metadata_column] == category
            color = colors[i]
            
            plt.scatter(Y[mask, 0], Y[mask, 1], color=color, label=category, s=dot_size)
        
        plt.legend(title=metadata_column, markerscale=5, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def pairwise_comparison(original, reconstructed, sample_names=None):
    """
    Plots multiple subplots, each containing two lines that represent gene expression samples. If N > 8, only the first 8 samples will be plotted
    :param original: NxD np array or pd DataFrame of original gene expression data
    :param reconstructed: NxD np array or pd DataFrame of reconstructed gene expression data
    :param title: A optional string that contains the plot title
    :param sample_names: a list of strings with the title of each sample. len(sample_names) should be 8
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
    plt.show()


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

