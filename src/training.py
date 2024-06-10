import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import r2_score as r2_score_sk
from torcheval.metrics.functional import r2_score as r2_score_torch

from src.models.fc import VAE


# loss function
def variational_loss(x_recon, x, mu, sigma, beta):
    """
    calculates the MSE loss and KLD and sums them up weighted with beta (mse + beta * kld)
    :param x_recon: reconstruction data
    :param x: original data
    :param mu: Mean
    :param sigma: Standard deviation
    :param beta: weight of KLD
    :return: the summed loss, MSE and KLD
    """
    mse = F.mse_loss(x_recon, x, reduction="mean")
    kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    loss = mse + beta * kld
    return loss, mse, kld

# training function
def variational_train(model, train_loader, val_loader, optimizer, scheduler,
                      writer, writer_val, epoch, device, beta=0.0002):
    """
    Trains a variational autoencoder.

    :param model: The model used for training.
    :param train_loader: The DataLoader of the train set.
    :param val_loader: The DataLoader of the validation set.
    :param optimizer: The optimizer.
    :param scheduler: The scheduler changing the learning rate.
    :param writer: Tensorboard logger for training.
    :param writer_val: Tensorboard logger for validation.
    :param epoch: Integer value of the current epoch.
    :param device: The device to use.
    """

    model.train()

    num_log_steps = 100  # Define the number of log steps per epoch

    for train_iteration, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch + 1}")):
        optimizer.zero_grad()

        batch = batch.to(device)
        batch_recon, mu, sigma = model(batch)

        loss, mse, kld = variational_loss(batch_recon, batch, mu, sigma, beta)

        loss.backward()
        optimizer.step()

        r2 = r2_score(batch.detach(), batch_recon.detach())


        current_step = (epoch * num_log_steps +
                        (train_iteration * num_log_steps) // len(train_loader))

        variational_logging(writer, "vae", loss.item(), mse.item(), kld.item(),
                            r2, current_step)

    scheduler.step()

    model.eval()

    with torch.no_grad():
        for val_iteration, batch in enumerate(tqdm(val_loader, desc=f"Validation epoch {epoch + 1}")):

            batch = batch.to(device)
            batch_recon, mu, sigma = model(batch)

            loss, mse, kld = variational_loss(batch_recon, batch, mu, sigma, beta)

            r2 = r2_score(batch.detach(), batch_recon.detach())


            current_step = epoch * num_log_steps + (val_iteration * num_log_steps) // len(val_loader)

            variational_logging(writer_val, "vae", loss.item(),
                                mse.item(), kld.item(), r2,
                                current_step)


def variational_logging(writer, model_name, loss, mse, kld, r2,
                        current_step):
    # when avg is used for mse calculation, no rescaling is required. Otherwise: e.g. loss / batch_size
    loss_scaled = loss
    mse_scaled = mse
    kld_scaled = kld

    writer.add_scalar(f"{model_name} loss combined", loss_scaled, current_step)
    writer.add_scalar(f"{model_name} mse loss", mse_scaled, current_step)
    writer.add_scalar(f"{model_name} kld loss", kld_scaled, current_step)
    writer.add_scalar(f"{model_name} R2 Score", r2, current_step)


def r2_score(y, y_pred):
    """Compute the R² coefficient"""
    residual = torch.sum((y - y_pred) ** 2)
    total = torch.sum((y - torch.mean(y)) ** 2)
    r2 = 1 - (residual / total)
    return r2

# for training without tensorboard logging (currently used in main.py):
num_epochs = 10
input_dim = 5000
hidden_one_dim = 2048
hidden_two_dim = 512
latent_dim = 128
learning_rate = 0.0001
beta = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(input_dim, hidden_one_dim, hidden_two_dim, latent_dim).to(device)
criterion = variational_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def epochs_loop(train_loader, val_loader):
    metrics = {
        "train_loss": [],
        "train_R2": [],
        "val_loss": [],
        "val_R2": []
    }

    for epoch in range(num_epochs):
        train_loss, train_R2 = batch_train_loop(train_loader)
        val_loss, val_R2 = batch_val_loop(val_loader)

        metrics["train_loss"].append(train_loss)
        metrics["train_R2"].append(train_R2)
        metrics["val_loss"].append(val_loss)
        metrics["val_R2"].append(val_R2)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Train R2: {train_R2:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val R2: {val_R2:.4f}")

    return metrics


def batch_train_loop(train_loader):
    model.train()
    total_loss = 0
    total_r2 = 0
    count = 0

    for batch in train_loader:
        optimizer.zero_grad()

        batch = batch.to(device)

        outputs, mu, sigma = model(batch)
        loss, _, _ = criterion(outputs, batch, mu, sigma, beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_r2 += r2_score(batch.detach().cpu(), outputs.detach().cpu())
        count += 1

    return total_loss / count, total_r2 / count


def batch_val_loop(val_loader):
    model.eval()
    total_loss = 0
    total_r2 = 0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs, mu, sigma = model(batch)
            loss, _, _ = criterion(outputs, batch, mu, sigma, beta)

            total_loss += loss.item()
            total_r2 += r2_score(batch.detach().cpu(), outputs.detach().cpu())
            count += 1

    return total_loss / count, total_r2 / count
