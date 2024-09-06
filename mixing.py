import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.loader import load_data


def generate_mixed_data(A, B, genes, meta, number_samples, filter_backbone, filter_auxiliary):
    """
    generates artificial gene expression samples and their corresponding metadata by mixing with backbone
    :param A: parameter a of the Beta distribution
    :param B: parameter b of the Beta distribution
    :param genes: gene expression dataframe
    :param meta: metadata dataframe
    :param number_samples: the number of samples to generate
    :param filter_backbone: filter for backbone data
    :param filter_auxiliary: filter for auxiliary data
    :return: two dataframes for gene expression values and metadata
    """

    genes_backbone = genes[filter_backbone]
    genes_auxiliary = genes[filter_auxiliary]
    meta_backbone = meta[filter_backbone]
    meta_auxiliary = meta[filter_auxiliary]

    meta_columns = meta.columns

    # combine the gene expression data with its corresponding metadata and randomly sample the gene expression data
    combined_backbone = genes_backbone.join(meta_backbone, how='inner')
    combined_auxiliary = genes_auxiliary.join(meta_auxiliary, how='inner')

    if len(combined_backbone) != len(meta_backbone) or len(combined_auxiliary) != len(meta_auxiliary):
        sys.exit("Samples in gene expression and meta do not match")

    sampled_auxiliary = combined_auxiliary.sample(n=number_samples, random_state=0,
                                                  replace=True)
    sampled_backbone = combined_backbone.sample(n=number_samples, random_state=0,
                                                    replace=True)

    sampled_auxiliary_meta = sampled_auxiliary[meta_columns]
    sampled_auxiliary_genes = sampled_auxiliary.drop(meta_columns, axis=1)
    sampled_backbone_meta = sampled_backbone[meta_columns]
    sampled_backbone_genes = sampled_backbone.drop(meta_columns, axis=1)

    # generate the weight vector using a beta distribution and flipping values below 0.5 to ensure the backbone to
    # always be the majority class
    beta_vector = np.random.beta(A, B, size=number_samples)
    to_flip = beta_vector < 0.5
    beta_vector[to_flip] = 1 - beta_vector[to_flip]
    scaling_backbone = beta_vector
    scaling_auxiliary = 1 - beta_vector

    # multiply samples by their random weights
    sampled_auxiliary_genes = sampled_auxiliary_genes.mul(scaling_auxiliary, axis=0)
    sampled_backbone_genes = sampled_backbone_genes.mul(scaling_backbone, axis=0)

    sampled_auxiliary_genes.reset_index(drop=True, inplace=True)
    sampled_backbone_genes.reset_index(drop=True, inplace=True)

    # add the minority (sampled_auxiliary_genes) to the backbone (sampled_backbone_genes)
    mixed_genes = sampled_auxiliary_genes + sampled_backbone_genes

    # combine the purity metadata
    high_purity_scaled = sampled_auxiliary_meta["cancer_purity"].mul(scaling_auxiliary).reset_index(drop=True)
    low_purity_scaled = sampled_backbone_meta["cancer_purity"].mul(scaling_backbone).reset_index(drop=True)
    mixed_purity = high_purity_scaled.add(low_purity_scaled)

    # define additional metadata information
    disease_backbone = sampled_backbone_meta["diagnosis"].reset_index(drop=True)
    disease_auxiliary = sampled_auxiliary_meta["diagnosis"].reset_index(drop=True)
    dataset_backbone = sampled_backbone_meta["dataset"].reset_index(drop=True)
    dataset_auxiliary = sampled_auxiliary_meta["dataset"].reset_index(drop=True)

    mixed_meta = pd.DataFrame({
        "cancer_purity": mixed_purity,
        "names_backbone": sampled_backbone_meta.index,
        "names_auxiliary": sampled_auxiliary_meta.index,
        "disease_backbone": disease_backbone,
        "disease_auxiliary": disease_auxiliary,
        "weight_backbone": scaling_backbone,
        "weight_auxiliary": scaling_auxiliary,
        "dataset_backbone": dataset_backbone,
        "dataset_auxiliary": dataset_auxiliary,
        "is_mixed": pd.Series(True, index=dataset_auxiliary.index)
    })

    return mixed_genes, mixed_meta


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


def main():
    if len(sys.argv) < 8 & len(sys.argv) > 9:
        sys.exit("Usage: python3 mixing.py <path/to/ccle.pickle> <path/to/tcga.pickle> <path/to/output/> "
                 "<number mixed samples per cancer type> <A> <B> <Include non-mixed data | True or False> "
                 "[Optional: <list of diseases, e.g. \"Disease1 Disease2\">]")
    ccle_pickle_path = sys.argv[1]
    tcga_pickle_path = sys.argv[2]
    output_dir = sys.argv[3]
    number_generated = int(sys.argv[4])
    A = float(sys.argv[5])
    B = float(sys.argv[6])
    include_original_data = sys.argv[7].lower() == 'true'
    # Check if the optional list of diseases is provided
    provided_diseases_exists = len(sys.argv) == 9
    if provided_diseases_exists:
        provided_diseases = sys.argv[8].split()
    else:
        provided_diseases = None

    genes, meta = load_data(ccle_pickle_path, tcga_pickle_path)

    train_genes, val_genes, train_meta, val_meta = train_test_split(genes, meta, test_size=0.2, random_state=42)

    all_diseases = train_meta["diagnosis"].unique()

    if provided_diseases is None:
        provided_diseases = all_diseases
    mixed_genes_list = []
    mixed_meta_list = []

    healthy_limit = 0.2

    # mix for the provided cancer types
    for cancer_type in provided_diseases:
        # define filters that will be used for generating samples with different backbones and auxiliary samples
        cancer_type_filter = train_meta["diagnosis"] == cancer_type
        dummy_filter = pd.Series(True, index=train_meta.index)
        healthy_filter = train_meta["cancer_purity"] < healthy_limit
        tcga_filter = train_meta["dataset"] == "tcga"
        ccle_filter = train_meta["dataset"] == "ccle"
        ccle_cancer_type_filter = ccle_filter & cancer_type_filter
        healthy_cancer_type_filter = healthy_filter & cancer_type_filter

        # count how many samples exist for different filters
        number_ccle_cancer_type_samples = ccle_cancer_type_filter.sum()
        number_healthy_samples = healthy_cancer_type_filter.sum()

        if number_healthy_samples > 0:
            # mix healthy (<healthy_limit) as backbone with TCGA
            healthy_tcga_mixed_genes, healthy_tcga_mixed_meta = generate_mixed_data(A, B, train_genes, train_meta,
                                                                                     number_generated,
                                                                                     healthy_cancer_type_filter,
                                                                                     tcga_filter)
            mixed_genes_list.append(healthy_tcga_mixed_genes)
            mixed_meta_list.append(healthy_tcga_mixed_meta)

            # mix healthy (<healthy_limit) as backbone with CCLE
            healthy_ccle_mixed_genes, healthy_ccle_mixed_meta = generate_mixed_data(A, B, train_genes, train_meta,
                                                                                    number_generated,
                                                                                    healthy_cancer_type_filter,
                                                                                    ccle_filter)
            mixed_genes_list.append(healthy_ccle_mixed_genes)
            mixed_meta_list.append(healthy_ccle_mixed_meta)

        # mix with ccle as backbone
        if number_ccle_cancer_type_samples > 0:
            ccle_mixed_genes, ccle_mixed_meta = generate_mixed_data(A, B, train_genes, train_meta, number_generated,
                                                                ccle_cancer_type_filter, dummy_filter)
            mixed_genes_list.append(ccle_mixed_genes)
            mixed_meta_list.append(ccle_mixed_meta)

        # mix any samples of the current disease with any
        cancer_type_genes, cancer_type_meta = generate_mixed_data(A, B, train_genes, train_meta, number_generated,
                                                                  cancer_type_filter, dummy_filter)
        mixed_genes_list.append(cancer_type_genes)
        mixed_meta_list.append(cancer_type_meta)


    mixed_genes = pd.concat(mixed_genes_list, ignore_index=True)
    mixed_meta = pd.concat(mixed_meta_list, ignore_index=True)

    if include_original_data:
        # filter for the diseases in the given provided_diseases
        diseases_filter_train = train_meta["diagnosis"].isin(provided_diseases)
        diseases_filter_val = val_meta["diagnosis"].isin(provided_diseases)

        val_genes = val_genes[diseases_filter_val]
        val_meta = val_meta[diseases_filter_val]

        train_diseases_genes = train_genes[diseases_filter_train]
        mod_meta = modified_meta(train_meta)
        train_diseases_meta = mod_meta[diseases_filter_train]

        # combine the mixed and original data, only including diseases in provided_diseases
        mixed_genes = pd.concat([mixed_genes, train_diseases_genes], ignore_index=True)
        mixed_meta = pd.concat([mixed_meta, train_diseases_meta], ignore_index=True)

    # generate the file name
    if provided_diseases_exists:
        diseases_str =  "_only_" + str(provided_diseases)
    else:
        diseases_str = ""
    file_name = (f"mixed_genes_num_samples={number_generated}_a={A}_b={B}"
                 f"_contains_original={include_original_data}{diseases_str}.pickle")

    # save the file
    with open(output_dir + file_name, "wb") as handle:
        pickle.dump(
            {
                "rnaseq": mixed_genes,
                "meta": mixed_meta,
                "rnaseq_val": val_genes,
                "meta_val": val_meta
            },
            handle)


if __name__ == "__main__":
    main()
