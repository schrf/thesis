import datetime
import os

import torch
from torch import optim
from torch.nn import L1Loss
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from src.data_visualization import plot_results

from src.models.ResNet import MultiTaskVAE

# initializations for the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
input_dim = 3350
latent_dim = 64
lr = 0.0001
lr_scaling = lr
kl_loss_weight = 0.00001
model = MultiTaskVAE(input_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# initializations required for GradNorm
Weightloss1 = torch.tensor([1.0], requires_grad=True, device=device)
Weightloss2 = torch.tensor([1.0], requires_grad=True, device=device)
params = [Weightloss1, Weightloss2]
scaling_optimizer = optim.Adam(params, lr=lr_scaling)
Gradloss = L1Loss()
alpha = 0.12
l0_recon = None
l0_purity = None


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


def epochs_loop(train_loader, val_loader, train_set, val_set, plot_comment):
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

    print(f"{device} is used")

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%m-%d_%H-%M")
    if plot_comment != "":
        plot_comment = " " + plot_comment
    plot_dir = "plots/multitask/" + formatted_time + plot_comment

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    save_metadata(plot_dir)

    for epoch in range(num_epochs):
        train_loss, train_recon, train_reg, train_kl, train_R2, w1, w2 = batch_train_loop(train_loader, epoch)
        val_loss, val_recon, val_reg, val_kl, val_R2 = batch_val_loop(val_loader)

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

    return metrics, model


def batch_train_loop(train_loader, epoch):
    global params
    global l0_recon
    global l0_purity
    model.train()
    total_loss = 0
    total_r2 = 0
    total_kl_loss = 0
    total_reg_loss = 0
    total_recon_loss = 0
    total_w1 = 0
    total_w2 = 0
    count = len(train_loader)

    for iter, batch in enumerate(train_loader):

        x, w, _ = batch
        x, w = x.to(device), w.to(device)

        x_hat, w_hat, mu, sigma = model(x)
        recon_loss = params[0] * MSE(x_hat, x)
        purity_loss = params[1] * MSE(w_hat, w)
        kl_loss = KLD(mu, sigma) * kl_loss_weight
        loss = torch.div(torch.add(recon_loss, purity_loss), 2) + kl_loss

        # for the first epoch with no l0
        if epoch == 0:
            l0_recon = recon_loss.data
            l0_purity = purity_loss.data

        optimizer.zero_grad()

        loss.backward(retain_graph=True)

        last_common_layer = model.encoder.fc.weight

        #TODO: check why G1 and G2 are Python floats instead of pytorch tensors and if the warning
        # "Using a target size (torch.Size([1])) that is different to the input size (torch.Size([]))" is a problem

        # Getting gradients of the first layers of each task and calculate the gradients l2-norm
        G1R = torch.autograd.grad(recon_loss, last_common_layer, retain_graph=True, create_graph=True)
        G1 = torch.norm(G1R[0], 2)
        G2R = torch.autograd.grad(purity_loss, last_common_layer, retain_graph=True, create_graph=True)
        G2 = torch.norm(G2R[0], 2)
        G_avg = torch.div(torch.add(G1, G2), 2)

        # Calculating relative losses
        lhat1 = torch.div(recon_loss, l0_recon)
        lhat2 = torch.div(purity_loss, l0_purity)
        lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)

        # Calculating relative inverse training rates for tasks
        inv_rate1 = torch.div(lhat1,lhat_avg)
        inv_rate2 = torch.div(lhat2,lhat_avg)

        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C1 = G_avg*(inv_rate1)**alpha
        C2 = G_avg*(inv_rate2)**alpha
        C1 = C1.detach()
        C2 = C2.detach()

        scaling_optimizer.zero_grad()

        # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
        Lgrad = torch.add(Gradloss(G1, C1),Gradloss(G2, C2))
        Lgrad.backward()

        # Updating loss weights
        scaling_optimizer.step()

        # Updating model weights
        optimizer.step()

        # Renormalizing the losses weights
        coef = 2/torch.add(Weightloss1, Weightloss2)
        params = [coef*Weightloss1, coef*Weightloss2]


        total_loss += loss.item() / count
        total_kl_loss += kl_loss.item() / count
        total_reg_loss += purity_loss.item() / count
        total_recon_loss += recon_loss.item() / count
        total_r2 += r2_score(x.detach().cpu(), x_hat.detach().cpu()) / count
        total_w1 += float(params[0].detach().cpu()) / count
        total_w2 += float(params[1].detach().cpu()) / count


    return total_loss, total_recon_loss, total_reg_loss, total_kl_loss, total_r2, total_w1, total_w2

# TODO: check why batch losses seem to be twice as large as train losses

def batch_val_loop(val_loader):
    model.eval()
    total_loss = 0
    total_r2 = 0
    total_kl_loss = 0
    total_reg_loss = 0
    total_recon_loss = 0
    count = len(val_loader)
    recon_loss_weight = float(params[0].detach().cpu())
    reg_head_weight = float(params[1].detach().cpu())

    with torch.no_grad():
        for batch in val_loader:
            x, w, _ = batch
            x, w = x.to(device), w.to(device)

            x_hat, w_hat, mu, sigma = model(x)
            recon_loss = recon_loss_weight * MSE(x_hat, x)
            purity_loss = reg_head_weight * MSE(w_hat, w)
            kl_loss = KLD(mu, sigma) * kl_loss_weight
            loss = torch.div(torch.add(recon_loss, purity_loss), 2) + kl_loss

            total_loss += loss.item() / count
            total_kl_loss += kl_loss.item() / count
            total_reg_loss += purity_loss.item() / count
            total_recon_loss += recon_loss.item() / count
            total_r2 += r2_score(x.detach().cpu(), x_hat.detach().cpu()) / count

    return (total_loss, total_recon_loss, total_reg_loss,
            total_kl_loss, total_r2)

def save_metadata(plot_dir):
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