from pacmap import PaCMAP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def combined_data_pacmap(data) -> None:
    # Step 1: Drop NaN values and normalize the age column
    data.dropna(inplace=True)
    data['normalized_age'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())

    # Extract the data columns for Pacmap visualization
    data_columns = [col for col in data.columns if col not in ['primary_disease', 'gender', 'age', 'dataset', 'normalized_age']]

    # Set the dot size for the plot
    dot_size = 1

    # Initialize Pacmap
    model = PaCMAP()

    # select data_columns for all genes or top_5000_columns for only the 5000 most variant:
    variances = data[data_columns].var()

    # Sort variances in descending order and select the top 5000 columns
    top_5000_columns = variances.sort_values(ascending=False).head(5000).index

    # Fit the model (select top_5000_columns for only 5000 most variant, or data_columns for all data)
    Y = model.fit_transform(data[data_columns])

    # Visualization 1: Primary Disease
    create_pacmap_visualization('primary_disease', 'Pacmap Visualization 5000 most variant genes: Primary Disease')

    # Visualization 2: Gender
    create_pacmap_visualization('gender', 'Pacmap Visualization 5000 most variant genes: Gender')

    # Visualization 3: Age
    create_pacmap_visualization('age', 'Pacmap Visualization 5000 most variant genes: Age')

    # Visualization 4: Dataset
    create_pacmap_visualization('dataset', 'Pacmap Visualization 5000 most variant genes: Dataset')

# Helper function to create Pacmap visualization
def create_pacmap_visualization(metadata_column, title):
    
    # Plot the Pacmap visualization
    plt.figure(figsize=(8, 6))
    
    # Color coding
    if metadata_column == 'age':
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=data['normalized_age'], cmap='viridis', s=dot_size)
        plt.colorbar(scatter, label='Relative Age')
    else:
        categories = data[metadata_column].unique()
        num_categories = len(categories)
        
        # Generate distinct colors for each category
        colors = [mcolors.to_rgba(f'C{i}') for i in range(num_categories)]
        
        for i, category in enumerate(categories):
            mask = data[metadata_column] == category
            plt.scatter(Y[mask, 0], Y[mask, 1], color=colors[i], label=category, s=dot_size)
        plt.legend(title=metadata_column, markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
