from pandas import DataFrame


def z_score_normalization_rowwise(df, filter):
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


def z_score_normalization_columnwise(df, filter):
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


def filter_variance(df: DataFrame, filter: int):
    """
    lists the columns of the most variant genes
    :param df: the gene expression dataframe
    :param filter: integer value indicating how many genes should be maintained
    :return: the column names of the most variant genes
    """
    variances = df.var()

    # Sort variances in descending order and select the most variant columns
    selected_columns = variances.sort_values(ascending=False).head(filter).index

    return selected_columns
