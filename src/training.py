import datetime
import os
import pathlib
import pickle

import torch
import torch.nn.functional as F

from src.data_visualization import plot_results
from torchinfo import summary

# loss functions
def KLD(mu, sigma):
    """calculate KL-divergence"""
    kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return kld


def MSE(x_recon, x):
    """calculate MSE. If x contains NaN, then it will be masked and the loss
    will only be calculated based on other values"""
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


def epochs_loop(model, optimizer, scheduler, train_set, val_set, hyperparams, device, comment):
    """executes all epochs and trains, validates and saves the results"""
    metrics = {
        "train_loss": [],
        "train_R2": [],
        "val_loss": [],
        "val_R2": [],
        "train_kl": [],
        "val_kl": [],
        "train_reg": [],
        "val_reg": [],
        "train_recon": [],
        "val_recon": [],
        "w1": [],
        "w2": []
    }

    tasks_weights, kl_loss_weight, batch_size, num_epochs = (hyperparams["task_weights"],
                                                                     hyperparams["kl_loss_weight"],
                                                                     hyperparams["batch_size"],
                                                                     hyperparams["num_epochs"])

    print(f"{device} is used")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m-%d_%H-%M")
    if comment != "":
        comment = " " + comment
    plot_dir = "plots/multitask/" + formatted_time + comment
    model_dir = "trained_models/" + formatted_time + comment
    r2_model_file = None
    loss_model_file = None

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_metadata(plot_dir, num_epochs, hyperparams["input_dim"], hyperparams["latent_dim"], hyperparams["lr"],
                  hyperparams["lr_scaling"], hyperparams["kl_loss_weight"], hyperparams["alpha"])

    save_model_architecture(model_dir, model)

    for epoch in range(num_epochs):
        train_loss, train_recon, train_reg, train_kl, train_R2, w1, w2 = train_loop(model, train_loader, tasks_weights, kl_loss_weight, optimizer, device)
        val_loss, val_recon, val_reg, val_kl, val_R2 = val_loop(model, val_loader, tasks_weights, kl_loss_weight, device)

        if epoch % 20 == 0 & epoch != 0:
            scheduler.step()

        metrics["train_loss"].append(train_loss)
        metrics["train_recon"].append(train_recon)
        metrics["train_kl"].append(train_kl)
        metrics["train_reg"].append(train_reg)
        metrics["train_R2"].append(train_R2)
        metrics["val_loss"].append(val_loss)
        metrics["val_recon"].append(val_recon)
        metrics["val_kl"].append(val_kl)
        metrics["val_reg"].append(val_reg)
        metrics["val_R2"].append(val_R2)
        metrics["w1"].append(w1)
        metrics["w2"].append(w2)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Train R2: {train_R2:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val R2: {val_R2:.4f}")

        plot_results(metrics, model, train_set, val_set, plot_dir)

        with open(model_dir + "/metrics.pickle", "wb") as handle:
            pickle.dump(metrics, handle)


        # save a model if it has the best R2 score or the lowest loss
        if is_highest_score(metrics["val_R2"]):
            if r2_model_file is not None:
                old_file = pathlib.Path(r2_model_file)
                old_file.unlink()
            r2_model_file = model_dir + f"/best_r2_model_{metrics['val_R2'][-1]}.pt"
            torch.save(model, r2_model_file)
        if is_lowest_score(metrics["val_loss"]):
            if loss_model_file is not None:
                old_file = pathlib.Path(loss_model_file)
                old_file.unlink()
            loss_model_file = model_dir + f"/best_loss_model_{metrics['val_loss'][-1]}.pt"
            torch.save(model, loss_model_file)


def train_loop(model, train_loader, tasks_weights, kl_loss_weight, optimizer, device):
    """performs the training loop over all batches in the train dataloader once and optimizes the model parameters"""
    model.train()
    total_loss = 0
    total_r2 = 0
    total_kl_loss = 0
    total_reg_loss = 0
    total_recon_loss = 0
    total_w1 = 0
    total_w2 = 0
    count = len(train_loader)
    recon_loss_weight, purity_loss_weight = tasks_weights

    for iter, batch in enumerate(train_loader):

        x, w, _ = batch
        x, w = x.to(device), w.to(device)

        x_hat, w_hat, mu, sigma = model(x)
        recon_loss = recon_loss_weight * MSE(x_hat, x)
        purity_loss = purity_loss_weight * MSE(w_hat, w)
        kl_loss = KLD(mu, sigma) * kl_loss_weight
        loss = torch.div(torch.add(recon_loss, purity_loss), 2) + kl_loss

        optimizer.zero_grad()

        loss.backward()

        # Updating model weights
        optimizer.step()


        total_loss += loss.item() / count
        total_kl_loss += kl_loss.item() / count
        total_reg_loss += purity_loss.item() / count
        total_recon_loss += recon_loss.item() / count
        total_r2 += float(r2_score(x.detach().cpu(), x_hat.detach().cpu())) / count
        total_w1 += float(recon_loss_weight) / count
        total_w2 += float(purity_loss_weight) / count


    return total_loss, total_recon_loss, total_reg_loss, total_kl_loss, total_r2, total_w1, total_w2

def val_loop(model, val_loader, tasks_weights, kl_loss_weight, device):
    """performs the validation loop over all batches in the validation dataloader once"""
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_kl_loss = 0
    total_reg_loss = 0
    total_recon_loss = 0
    count = len(val_loader)
    recon_loss_weight, purity_loss_weight = tasks_weights

    with torch.no_grad():
        for batch in val_loader:
            x, w, _ = batch
            x, w = x.to(device), w.to(device)

            x_hat, w_hat, mu, sigma = model(x)
            recon_loss = recon_loss_weight * MSE(x_hat, x)
            purity_loss = purity_loss_weight * MSE(w_hat, w)
            kl_loss = KLD(mu, sigma) * kl_loss_weight
            loss = torch.div(torch.add(recon_loss, purity_loss), 2) + kl_loss

            total_loss += loss.item() / count
            total_kl_loss += kl_loss.item() / count
            total_reg_loss += purity_loss.item() / count
            total_recon_loss += recon_loss.item() / count
            total_r2 += float(r2_score(x.detach().cpu(), x_hat.detach().cpu())) / count

    return (total_loss, total_recon_loss, total_reg_loss,
            total_kl_loss, total_r2)

def save_metadata(plot_dir, num_epochs, input_dim, latent_dim, lr, lr_scaling, kl_loss_weight, alpha):
    with open(plot_dir + '/metadata.txt', 'w') as f:
        data = f"""num_epochs = {num_epochs}
                input_dim = {input_dim}
                latent_dim = {latent_dim}
                lr = {lr}
                lr_scaling = {lr_scaling}
                kl_loss_weight = {kl_loss_weight}
                alpha = {alpha}
                """
        f.write(data)

def save_model_architecture(model_dir, model):
    if not os.path.isfile(model_dir + "/summary.txt"):
        with open(model_dir + "/summary.txt", "w") as f:
            summary_str = str(summary(model, (1, 3350), verbose=0))
            f.write(summary_str)

def is_highest_score(list):
    """returns True if the last element in the list has the highest value, else False"""
    return list[-1] == max(list)

def is_lowest_score(list):
    """returns True if the last element in the list has the lowest value, else False"""
    return list[-1] == min(list)