import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18

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

class ResNet1DEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.in_channels, 1500)
        self.fc2 = nn.Linear(1500, 500)
        self.fc3 = nn.Linear(500, 200)
        self.mu = nn.Linear(200, out_features=latent_dim)
        self.sigma = nn.Linear(200, out_features=latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma
# Variational Sampling
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.fc(z)

# Regression Head
class RegressionHead(nn.Module):
    def __init__(self, latent_dim):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.fc(z)

# Full Model
class MultiTaskVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MultiTaskVAE, self).__init__()
        self.encoder = ResNet1DEncoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.regression_head = RegressionHead(latent_dim)

    def forward(self, x):
        mean, sigma = self.encoder(x)
        z = reparameterize(mean, sigma)
        recon_x = self.decoder(z)
        reg_y = self.regression_head(z)
        return recon_x, reg_y, mean, sigma
