import pickle
import sys
import numpy as np
import pandas as pd

from src.data_loader import load_data


def generate_mixed_data(A, B, genes, meta, generated_samples):
    low_purity = meta["cancer_purity"] < 0.6
    high_purity = meta["cancer_purity"] >= 0.6
    high_purity_genes = genes[high_purity]
    low_purity_genes = genes[low_purity]
    high_purity_meta = meta[high_purity]
    low_purity_meta = meta[low_purity]
    meta_columns = meta.columns
    high_purity_combined = high_purity_genes.join(high_purity_meta, how='inner')
    low_purity_combined = low_purity_genes.join(low_purity_meta, how='inner')
    if len(low_purity_combined) != len(low_purity_meta) or len(high_purity_combined) != len(high_purity_meta):
        sys.exit("Samples in gene expression and meta do not match")
    sampled_high_purity = high_purity_combined.sample(n=generated_samples, random_state=0,
                                                      replace=True)
    sampled_low_purity = low_purity_combined.sample(n=generated_samples, random_state=0,
                                                    replace=True)
    sampled_high_purity_meta = sampled_high_purity[meta_columns]
    sampled_high_purity_genes = sampled_high_purity.drop(meta_columns, axis=1)
    sampled_low_purity_meta = sampled_low_purity[meta_columns]
    sampled_low_purity_genes = sampled_low_purity.drop(meta_columns, axis=1)
    beta_vector = np.random.beta(A, B, size=generated_samples)
    to_flip = beta_vector < 0.5
    beta_vector[to_flip] = 1 - beta_vector[to_flip]
    low_purity_scaling = beta_vector
    high_purity_scaling = 1 - beta_vector
    sampled_high_purity_genes = sampled_high_purity_genes.mul(high_purity_scaling, axis=0)
    sampled_low_purity_genes = sampled_low_purity_genes.mul(low_purity_scaling, axis=0)
    sampled_high_purity_genes.reset_index(drop=True, inplace=True)
    sampled_low_purity_genes.reset_index(drop=True, inplace=True)
    mixed_genes = sampled_high_purity_genes + sampled_low_purity_genes
    high_purity_scaled = sampled_high_purity_meta["cancer_purity"].mul(high_purity_scaling).reset_index(drop=True)
    low_purity_scaled = sampled_low_purity_meta["cancer_purity"].mul(low_purity_scaling).reset_index(drop=True)
    mixed_purity = high_purity_scaled.add(low_purity_scaled)
    disease_backbone = sampled_low_purity_meta["diagnosis"].reset_index(drop=True)
    disease_auxiliary = sampled_high_purity_meta["diagnosis"].reset_index(drop=True)
    dataset_backbone = sampled_low_purity_meta["dataset"].reset_index(drop=True)
    dataset_auxiliary = sampled_high_purity_meta["dataset"].reset_index(drop=True)
    mixed_meta = pd.DataFrame({
        "cancer_purity": mixed_purity,
        "names_backbone": sampled_low_purity_meta.index,
        "names_auxiliary": sampled_high_purity_meta.index,
        "disease_backbone": disease_backbone,
        "disease_auxiliary": disease_auxiliary,
        "weight_backbone": low_purity_scaling,
        "weight_auxiliary": high_purity_scaling,
        "dataset_backbone": dataset_backbone,
        "dataset_auxiliary": dataset_auxiliary,
        "is_mixed": pd.Series(True, index=dataset_auxiliary.index)
    })
    return mixed_genes, mixed_meta


def modified_meta(meta):
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
    if len(sys.argv) != 8:
        sys.exit("Usage: python3 data_transformation.py <path/to/ccle.pickle> <path/to/tcga.pickle> <path/to/output/> "
                 "<proportion mixed data> <A> <B> <Include non-mixed data | True or False>")
    ccle_pickle_path = sys.argv[1]
    tcga_pickle_path = sys.argv[2]
    output_dir = sys.argv[3]
    proportion_mixed = int(sys.argv[4])
    A = float(sys.argv[5])
    B = float(sys.argv[6])
    include_original_data = sys.argv[7].lower() == 'true'


    genes, meta = load_data(ccle_pickle_path, tcga_pickle_path)
    num_samples = len(genes)
    generated_samples = int(num_samples * proportion_mixed)
    mixed_genes, mixed_meta = generate_mixed_data(A, B, genes, meta, generated_samples)
    if include_original_data:
        mod_meta = modified_meta(meta)
        mixed_genes = pd.concat([mixed_genes, genes], ignore_index=True)
        mixed_meta = pd.concat([mixed_meta, mod_meta], ignore_index=True)



    with open(output_dir + f"mixed_genes_proportion={proportion_mixed}_a={A}_b={B}"
                           f"_contains_original={include_original_data}.pickle", "wb") as handle:
        pickle.dump(
            {
                "rnaseq": mixed_genes,
                "meta": mixed_meta
            },
            handle)



if __name__ == "__main__":
    main()