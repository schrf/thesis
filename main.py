import sys
import torch
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, load_mixed_data
from src.training import epochs_loop
from src.datasets.simple_dataset import SimpleDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch import optim

from src.models.ResNet import MultiTaskVAE

#hyperparameters
recon_loss_weight = 0.85
task_weights = [recon_loss_weight, 1 - recon_loss_weight]
hyperparams = {
    "num_epochs": 400,
    "input_dim": 3350,
    "latent_dim": 64,
    "lr": 0.0001,
    "lr_scaling": 0.0001,
    "batch_size": 64,
    "kl_loss_weight": 0.00001,
    "task_weights": task_weights,
    "alpha": None
}

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskVAE(hyperparams["input_dim"], hyperparams["latent_dim"]).to(device)
optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
scheduler = ExponentialLR(optimizer, gamma=0.98)


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
            train_genes, val_genes, train_meta, val_meta = train_test_split(genes, meta, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"An error occurred during loading ccle and tcga: {e}")
            sys.exit(1)
    else:
        mixed_path = sys.argv[1]
        comment = sys.argv[2]
        try:
            train_genes, val_genes, train_meta, val_meta = load_mixed_data(mixed_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"An error occurred during loading mixed data: {e}")
            sys.exit(1)

    transform = {
        "z_score": "per_sample"
    }

    train_set = SimpleDataset(train_genes, train_meta, transform=transform)
    val_set = SimpleDataset(val_genes, val_meta, transform=transform)

    epochs_loop(model, optimizer, scheduler, train_set, val_set, hyperparams, device, comment)


if __name__ == "__main__":
    main()
