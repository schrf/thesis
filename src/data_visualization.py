from pacmap import PaCMAP
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def combined_data_pacmap(data, viz_preprocessing):
    # Extract the data columns for Pacmap visualization
    data_columns = [col for col in data.columns if col not in ['primary_disease', 'gender', 'age', 'dataset', 'normalized_age']]

    if viz_preprocessing["z_score_norm"] == "per_gene":
        data = _z_score_normalization_columnwise(data, data_columns)
    elif viz_preprocessing["z_score_norm"] == "per_sample":
        data = _z_score_normalization_rowwise(data, data_columns)

    _combined_data_pacmap(data, viz_preprocessing, data_columns)


def _combined_data_pacmap(data, viz_preprocessing, data_columns):
    # Step 1: Drop NaN values and normalize the age column
    data.dropna(inplace=True)
    data['normalized_age'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())

    # Initialize Pacmap
    model = PaCMAP()

    if viz_preprocessing["only_most_variant"] != None:
        # select data_columns for all genes or top_5000_columns for only the 5000 most variant:
        variances = data[data_columns].var()

        # Sort variances in descending order and select the top 5000 columns
        selected_columns = variances.sort_values(ascending=False).head(viz_preprocessing["only_most_variant"]).index
    else:
        selected_columns = data_columns
    
    # Fit the model
    Y = model.fit_transform(data[selected_columns])

    # Visualizations
    create_pacmap_visualization(data, Y, 'primary_disease', 'Pacmap Visualization 5000 most variant genes: Primary Disease')
    create_pacmap_visualization(data, Y, 'gender', 'Pacmap Visualization 5000 most variant genes: Gender')
    create_pacmap_visualization(data, Y, 'age', 'Pacmap Visualization 5000 most variant genes: Age')
    create_pacmap_visualization(data, Y, 'dataset', 'Pacmap Visualization 5000 most variant genes: Dataset')

# Helper function to create Pacmap visualization
def create_pacmap_visualization(data, Y, metadata_column, title):
    
    # Plot the Pacmap visualization
    plt.figure(figsize=(8, 6))
    
    # Color coding
    if metadata_column == 'age':
        scatter = plt.scatter(Y[:, 0], Y[:, 1], c=data['normalized_age'], cmap='viridis', s=1)
        plt.colorbar(scatter, label='Relative Age')
    else:
        categories = data[metadata_column].unique()
        num_categories = len(categories)
        
        # Generate distinct colors for each category
        colors = [mcolors.to_rgba(f'C{i}') for i in range(num_categories)]
        
        for i, category in enumerate(categories):
            mask = data[metadata_column] == category
            plt.scatter(Y[mask, 0], Y[mask, 1], color=colors[i], label=category, s=1)
        plt.legend(title=metadata_column, markerscale=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def _z_score_normalization_rowwise(df, filter):
    # Calculate mean and standard deviation for each row
    filtered_df = df.loc[:, filter]

    # Calculate means and standard deviations for the filtered columns
    means = filtered_df.mean(axis=1)
    stds = filtered_df.std(axis=1)

    # Subtract means from the filtered DataFrame and divide by standard deviations
    normalized_filtered_df = filtered_df.sub(means, axis=0).div(stds, axis=0)

    # Merge normalized values back into the original DataFrame
    normalized_df = df.copy()  # Make a copy of the original DataFrame
    normalized_df.loc[:, filter] = normalized_filtered_df  # Update filtered columns with normalized values

    return normalized_df


def _z_score_normalization_columnwise(df, filter):
    # Filter DataFrame to include only the columns specified in the filter
    filtered_df = df.loc[:, filter]

    # Calculate mean and standard deviation for each column in the filtered DataFrame
    means = filtered_df.mean()
    stds = filtered_df.std()
    
    # Apply z-score normalization to each column in the filtered DataFrame
    normalized_filtered_df = (filtered_df - means) / stds
    
    # Merge normalized values back into the original DataFrame
    normalized_df = df.copy()  # Make a copy of the original DataFrame
    normalized_df.loc[:, filter] = normalized_filtered_df  # Update filtered columns with normalized values

    return normalized_df