import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import r2_score as r2_score_sk
from torcheval.metrics.functional import r2_score as r2_score_torch

from src.models.fc import VAE
from src.models.ResNet import MultiTaskVAE

# for training without tensorboard logging (currently used in main.py):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 60
input_dim = 3350
latent_dim = 64
model = MultiTaskVAE(input_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
kl_loss_weight = 0.1
reg_head_weight = 0.2
recon_loss_weight = 1 - kl_loss_weight - reg_head_weight


# loss functions
def KLD(mu, sigma):
    """calculate KL-divergence"""
    kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return kld


def MSE(x_recon, x):
    # Create a mask to identify non-NaN values in the target tensor
    #print(f"reconstructed shape: {x_recon}")
    #print(f"target shape: {x}")
    mask = ~torch.isnan(x)

    # Use the mask to filter out NaN values in both input and target tensors
    x_recon_filtered = x_recon[mask]
    x_filtered = x[mask]

    # Compute MSE only on the non-NaN values
    if x_filtered.numel() == 0:  # If no valid data points, return zero loss
        mse = torch.tensor(0.0, device=x.device)
    else:
        mse = F.mse_loss(x_recon_filtered, x_filtered, reduction="mean")

    return mse


def r2_score(y, y_pred):
    """Compute the R² coefficient"""
    residual = torch.sum((y - y_pred) ** 2)
    total = torch.sum((y - torch.mean(y)) ** 2)
    r2 = 1 - (residual / total)
    return r2


def epochs_loop(train_loader, val_loader):
    metrics = {
        "train_loss": [],
        "train_R2": [],
        "val_loss": [],
        "val_R2": []
    }

    print(f"{device} is used")

    for epoch in range(num_epochs):
        train_loss, train_R2 = batch_train_loop(train_loader)
        val_loss, val_R2 = batch_val_loop(val_loader)

        metrics["train_loss"].append(train_loss)
        metrics["train_R2"].append(train_R2)
        metrics["val_loss"].append(val_loss)
        metrics["val_R2"].append(val_R2)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Train R2: {train_R2:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val R2: {val_R2:.4f}")

    return metrics


def batch_train_loop(train_loader):
    model.train()
    total_loss = 0
    total_r2 = 0
    count = len(train_loader)

    for batch in train_loader:
        optimizer.zero_grad()

        x, w, _ = batch
        x, w = x.to(device), w.to(device)

        x_hat, w_hat, mu, sigma = model(x)
        recon_loss = MSE(x_hat, x)
        purity_loss = MSE(w_hat, w)
        kl_loss = KLD(mu, sigma)
        loss = recon_loss_weight * recon_loss + kl_loss_weight * kl_loss + reg_head_weight * purity_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_r2 += r2_score(x.detach().cpu(), x_hat.detach().cpu())

    return total_loss / count, total_r2 / count


def batch_val_loop(val_loader):
    model.eval()
    total_loss = 0
    total_r2 = 0
    count = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            x, w, _ = batch
            x, w = x.to(device), w.to(device)

            x_hat, w_hat, mu, sigma = model(x)
            recon_loss = MSE(x_hat, x)
            purity_loss = MSE(w_hat, w)
            kl_loss = KLD(mu, sigma)
            loss = recon_loss_weight * recon_loss + kl_loss_weight * kl_loss + reg_head_weight * purity_loss

            total_loss += loss.item()
            total_r2 += r2_score(x.detach().cpu(), x_hat.detach().cpu())

    return total_loss / count, total_r2 / count
