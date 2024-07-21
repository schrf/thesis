import pandas as pd
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

def combine_ccle_tcga(ccle_genes, ccle_meta, tcga_genes, tcga_meta):
    tcga_ccle_age = pd.concat([ccle_meta["age"], tcga_meta["cgc_case_age_at_diagnosis"]])

    tcga_ccle_diagnosis = pd.concat([ccle_meta["diagnosis"], tcga_meta["diagnosis"]])
    tcga_ccle_diagnosis = tcga_ccle_diagnosis.apply(lambda x: 'UNABLE TO CLASSIFY' if pd.isna(x) or x == '0' else x)

    tcga_ccle_gender = pd.concat([ccle_meta["sex"].str.lower(), tcga_meta["gdc_cases.demographic.gender"].fillna("unknown")])

    tcga_dataset = pd.Series(["tcga"] * len(tcga_meta), index=tcga_meta.index)
    ccle_dataset = pd.Series(["ccle"] * len(ccle_meta), index=ccle_meta.index)

    tcga_ccle_dataset = pd.concat([ccle_dataset, tcga_dataset])

    tcga_ccle_purity = pd.concat([pd.Series([100.0] * len(ccle_meta), index=ccle_meta.index, name="cancer_purity"), tcga_meta["tumor_percent"]])
    tcga_ccle_purity = tcga_ccle_purity / 100

    tcga_ccle_meta = pd.DataFrame({
        "dataset": tcga_ccle_dataset,
        "gender": tcga_ccle_gender,
        "diagnosis": tcga_ccle_diagnosis,
        "age": tcga_ccle_age,
        "cancer_purity": tcga_ccle_purity
    })

    tcga_ccle_genes = pd.concat([ccle_genes, tcga_genes])

    return tcga_ccle_genes, tcga_ccle_meta
