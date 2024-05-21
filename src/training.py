import torch
from tqdm import tqdm
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score
# loss function
def variational_loss_func(x_recon, x , mu, sigma, beta):
    mse = F.mse_loss(x_recon, x)
    kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    loss = mse + beta * kld
    return loss, mse, kld

# training function
def variational_train(model, train_loader, val_loader, optimizer, scheduler, writer, writer_val, epoch, device, beta=1.0):
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
        batch = batch.to(device)
        batch_recon, mu, sigma = model(batch)

        optimizer.zero_grad()

        loss, mse, kld = variational_loss_func(batch_recon, batch, mu, sigma, beta)

        loss.backward()
        optimizer.step()

        r2 = r2_score(batch_recon, batch)

        current_step = epoch * num_log_steps + (train_iteration * num_log_steps) // len(train_loader)
        writer.add_scalar("vae loss combined", loss.item(), current_step)
        writer.add_scalar("vae mse loss", mse.item(), current_step)
        writer.add_scalar("vae kld loss", kld.item(), current_step)
        writer.add_scalar("vae R2 Score", r2.item(), current_step)

    scheduler.step()

    model.eval()

    with torch.no_grad():
        for val_iteration, batch in enumerate(tqdm(val_loader, desc=f"Validation epoch {epoch + 1}")):
            batch = batch.to(device)
            batch_recon, mu, sigma = model(batch)

            loss, mse, kld = variational_loss_func(batch_recon, batch, mu, sigma, beta)

            r2 = r2_score(batch_recon, batch)

            current_step = epoch * num_log_steps + (val_iteration * num_log_steps) // len(val_loader)
            writer_val.add_scalar("vae loss combined", loss.item(), current_step)
            writer_val.add_scalar("vae mse loss", mse.item(), current_step)
            writer_val.add_scalar("vae kld loss", kld.item(), current_step)
            writer_val.add_scalar("vae R2 Score", r2.item(), current_step)
