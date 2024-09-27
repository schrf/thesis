import pandas as pd
import phate
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


def phate_transformation(data, verbose=1):
    """use PHATE to transform the data"""
    model = phate.PHATE(n_jobs=-2, verbose=verbose)
    return model.fit_transform(data)


def phate_per_disease(genes, meta):
    """
    takes the gene expression data and performs PHATE per disease. Combines the transformed data and its corresponding
    metadata to a dictionary
    :param genes: gene expression data
    :param meta: metadata with either a column named diagnosis or a column named disease_backbone
    :return: returns a dictionary like {"disease1": (PHATE_transformed_gene_expression, metadata), "disease2": ...}
    """
    genes = genes.sort_index()
    meta = meta.sort_index()

    # find the correct metadata column
    if "diagnosis" in meta.columns:
        disease_column_name = "diagnosis"
    elif "disease_backbone" in meta.columns:
        disease_column_name = "disease_backbone"
    else:
        raise Exception("No diagnosis column")

    cancer_types = meta[disease_column_name].unique()

    # create a dictionary that contains the phate-transformed gene expression and metadata for every disease
    diseases_dict = {}

    for cancer in cancer_types:
        filter = meta[disease_column_name] == cancer

        cancer_genes = genes[filter]
        cancer_meta = meta[filter]

        if len(cancer_genes) == 1:
            continue

        cancer_Y = phate_transformation(cancer_genes, verbose=0)

        diseases_dict[cancer] = cancer_Y, cancer_meta

    return diseases_dict


def modified_meta(meta):
    """takes a metadata dataframe
    returns a new one with backbone columns containing the original information and auxiliary columns containing None"""
    mod_meta = pd.DataFrame({
        "cancer_purity": meta["cancer_purity"],
        "names_backbone": meta.index,
        "names_auxiliary": pd.Series(None, index=meta.index),
        "disease_backbone": meta["diagnosis"],
        "disease_auxiliary": pd.Series(None, index=meta.index),
        "weight_backbone": pd.Series(1, index=meta.index),
        "weight_auxiliary": pd.Series(0, index=meta.index),
        "dataset_backbone": meta["dataset"],
        "dataset_auxiliary": pd.Series(None, index=meta.index),
        "is_mixed": pd.Series(False, index=meta.index)
    })
    return mod_meta
