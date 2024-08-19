import sys
import torch
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, load_mixed_data
from src.training import epochs_loop
from src.datasets.simple_dataset import SimpleDataset


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python main.py <ccle_path> <tcga_path> <comment for run> OR python main.py <mixed_path> <comment for run>")
        sys.exit(1)

    is_mixed = len(sys.argv) == 3
    if not is_mixed:
        ccle_path = sys.argv[1]
        tcga_path = sys.argv[2]
        comment = sys.argv[3]

        try:
            genes, meta = load_data(ccle_path, tcga_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"An error occurred during loading ccle and tcga: {e}")
            sys.exit(1)
    else:
        mixed_path = sys.argv[1]
        comment = sys.argv[2]
        try:
            genes, meta = load_mixed_data(mixed_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"An error occurred during loading mixed data: {e}")

    train_genes, val_genes, train_meta, val_meta = train_test_split(genes, meta, test_size=0.2, random_state=42)

    transform = {
        "z_score": "per_sample"
    }

    train_set = SimpleDataset(train_genes, train_meta, transform=transform)
    val_set = SimpleDataset(val_genes, val_meta, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

    metrics, model = epochs_loop(train_loader, val_loader, train_set, val_set, comment)


if __name__ == "__main__":
    main()
