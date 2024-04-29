import pandas as pd
import os.path


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
        print("faster loading of ccle and tcga and metadata")
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
