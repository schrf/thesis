import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder based on ResNet-18
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_length):
        super(Encoder, self).__init__()

        self.resblock1 = ResidualBlock1D(in_channels=1, out_channels=4)
        self.resblock2 = ResidualBlock1D(in_channels=4, out_channels=6)
        self.resblock3 = ResidualBlock1D(in_channels=6, out_channels=10)
        self.resblock4 = ResidualBlock1D(in_channels=10, out_channels=15)
        self.fc = nn.Linear(15 * input_length, 2 * latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.resblock1(x)
        x = self.resblock2(x)
        # x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.resblock3(x)
        x = self.resblock4(x)
        # x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        mean, logvar = torch.chunk(self.fc(x), 2, dim=1)
        return mean, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.fc = nn.Linear(latent_dim, output_dim * 15)
        self.conv1 = nn.ConvTranspose1d

    def forward(self, z):
        z = self.fc(z)
        return z

# Regression Head:
class RegressionHead(nn.Module):
    def __init__(self, latent_dim):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(latent_dim, int(latent_dim / 2))
        self.fc2 = nn.Linear(int(latent_dim / 2), int(latent_dim / 4))
        self.fc3 = nn.Linear(int(latent_dim / 4), 1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        return z

# Variational Sampling
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

# Full Model
class MultiTaskVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MultiTaskVAE, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim, input_length=input_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.regression_head = RegressionHead(latent_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = reparameterize(mean, logvar)
        recon_x = self.decoder(z)
        reg_y = self.regression_head(z)
        return recon_x, reg_y, mean, logvar
