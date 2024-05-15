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
    trains a variational autoencoder
    :param model: the model used for training
    :param train_loader: the dataloader of the train set
    :param optimizer: the optimizer
    :param scheduler: the scheduler changing the learning rate
    :param writer: tensorboard logger
    :param epoch: int value of the current epoch
    :param device: the device to use
    """

    model.train()

    for train_iteration, batch in enumerate(tqdm(train_loader, desc="Training epoch {}".format(epoch+1))):
        batch = batch.to(device)
        batch_recon, mu, sigma = model(batch)

        optimizer.zero_grad()

        loss, mse, kld = variational_loss_func(batch_recon, batch, mu, sigma, beta)

        loss.backward()
        optimizer.step()

        r2 = r2_score(batch_recon, batch)

        current_iteration = epoch * len(train_loader) + train_iteration
        writer.add_scalar("vae loss combined", loss.item(), current_iteration)
        writer.add_scalar("vae mse loss", mse.item(), current_iteration)
        writer.add_scalar("vae kld loss", kld.item(), current_iteration)
        writer.add_scalar("vae R2 Score", r2.item(), current_iteration)

    scheduler.step()

    model.eval()

    with torch.no_grad():
        for val_iteration, batch in enumerate(tqdm(val_loader, desc="Validation epoch {}".format(epoch+1))):
            batch = batch.to(device)
            batch_recon, mu, sigma = model(batch)

            loss, mse, kld = variational_loss_func(batch_recon, batch, mu , sigma, beta)

            r2 = r2_score(batch_recon, batch)

            current_iteration = epoch * len(train_loader) + val_iteration
            writer_val.add_scalar("vae loss combined", loss.item(), current_iteration)
            writer_val.add_scalar("vae mse loss", mse.item(), current_iteration)
            writer_val.add_scalar("vae kld loss", kld.item(), current_iteration)
            writer_val.add_scalar("vae R2 Score", r2.item(), current_iteration)
