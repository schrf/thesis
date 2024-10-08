import pickle
import pandas as pd
import os.path
import torch

from src.data_transformation import combine_ccle_tcga, modified_meta


def ccle_full_loader() -> pd.DataFrame:
    return pd.read_csv("data/CCLE_expression_full.csv", index_col=0)


def tcga_full_loader() -> pd.DataFrame:
    return pd.read_csv("data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena", sep="\t", index_col=0)


def ccle_tcga_loader() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    filter_list = pd.read_csv("data/filtered_17713_gene_names.csv")
    filter_list_stripped = filter_list["# Gene"].str.split(' \(').str[0]

    if not os.path.isfile("data/ccle.csv") or not os.path.isfile("data/tcga.csv"):
        print("load ccle from full file. Code is inefficient, this takes a while")
        ccle_full = ccle_full_loader()
        ccle_full.columns = ccle_full.columns.str.split(' \(').str[0]
        filtered_columns = [col for col in ccle_full.columns if filter_list_stripped.str.contains(col).any()]
        ccle = ccle_full[filtered_columns]
        ccle = ccle.sort_index()
        ccle = ccle.T.drop_duplicates(keep=False).T

        print("load tcga from full file. Code is inefficient, this takes a while")
        tcga_full = tcga_full_loader()
        tcga = tcga_full[tcga_full.index.isin(filter_list_stripped)]
        tcga = tcga.T
        tcga = tcga.sort_index()
        tcga = tcga.T.drop_duplicates(keep=False).T

        del filter_list
        del filter_list_stripped

        # only keep columns that are present in both filtered datasets
        common_columns = tcga.columns.intersection(ccle.columns)
        ccle = ccle[common_columns]
        tcga = tcga[common_columns]

        ccle_metadata = ccle_metadata_loader(ccle)

        tcga_metadata = tcga_metadata_loader()

        # reduce the tcga dataset to the samples that are also present in Diyuans reduced metadata set
        common_indices_tcga = tcga.index.intersection(tcga_metadata.index)
        tcga = tcga.loc[common_indices_tcga]

        tcga.to_csv("data/tcga.csv")
        ccle.to_csv("data/ccle.csv")

        return ccle, tcga, ccle_metadata, tcga_metadata
    else:
        print("fast loading of ccle and tcga and metadata as tcga.csv and ccle.csv are already present")
        print("load ccle")
        ccle = pd.read_csv("data/ccle.csv", index_col=0)
        print("load tcga")
        tcga = pd.read_csv("data/tcga.csv", index_col=0)
        print("load ccle_metadata")
        ccle_metadata = ccle_metadata_loader(ccle)
        print("load tcga_metadata")
        tcga_metadata = tcga_metadata_loader()
        return ccle, tcga, ccle_metadata, tcga_metadata


def ccle_metadata_loader(ccle: pd.DataFrame) -> pd.DataFrame:

    ccle_metadata_full = pd.read_csv("data/sample_info.csv")
    ccle_metadata = ccle_metadata_full[ccle_metadata_full["DepMap_ID"].isin(ccle.index)]
    ccle_metadata = ccle_metadata.set_index(ccle_metadata["DepMap_ID"])
    ccle_metadata = ccle_metadata.drop(columns="DepMap_ID")
    ccle_metadata = ccle_metadata.sort_index()
    return ccle_metadata


def tcga_metadata_loader() -> pd.DataFrame:
    tcga_metadata = pd.read_csv("data/final-resorted-samples-based-HiSeqV2-new.csv", sep=",", index_col=1, low_memory=False)
    tcga_metadata = tcga_metadata.drop(columns="Unnamed: 0")
    tcga_metadata.index = [index[:-1] for index in
                           tcga_metadata.index]  # remove the trailing letters, as they are not present in tcga indices
    return tcga_metadata


def load_data(ccle_path, tcga_path):
    """
    returns the combined ccle and tcga datasets
    :param ccle_path: path to ccle pickle file
    :param tcga_path: path to tcga pickle file
    :return: gene expression and metadata dataframe
    """

    # Open the pickle file in binary read mode
    with open(ccle_path, 'rb') as file:
        # Load the contents of the pickle file
        ccle = pickle.load(file)

    with open(tcga_path, 'rb') as file:
        tcga = pickle.load(file)

    ccle_genes = ccle["rnaseq"]
    ccle_meta = ccle["meta"]
    tcga_genes = tcga["rnaseq"]
    tcga_meta = tcga["meta"]
    del ccle, tcga

    combined_genes, combined_meta = combine_ccle_tcga(ccle_genes, ccle_meta, tcga_genes, tcga_meta)
    return combined_genes, combined_meta


def model_loader(files_list):
    """yields one model after the other from the given path string to the model files"""
    for file in files_list:
        model = torch.load(file, weights_only=False)
        yield model


def load_mixed_data(path, combine_train_val=False):
    """
    loads the data and returns gene expression values and the metadata, split into train and validation or both combined
    :param file_path: the file path to the pickle file containing the dictionary
    :param combine_train_val: whether to combine train and validation data or not
    :return: gene expression and metadata dataframe for each train and validation or for both combined
    """
    with open(path, "rb") as mixed:
        data = pickle.load(mixed)
    genes_mixed = data["rnaseq"]
    genes_val = data["rnaseq_val"]
    meta_mixed = data["meta"]
    meta_val = data["meta_val"]

    meta_val = modified_meta(meta_val)

    if combine_train_val:
        genes_mixed = pd.concat([genes_mixed, genes_val], ignore_index=True)
        meta_mixed = pd.concat([meta_mixed, meta_val], ignore_index=True)
        return genes_mixed, meta_mixed
    else:
        return genes_mixed, meta_mixed, genes_val, meta_val
