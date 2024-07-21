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
        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv1d(16, 24, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv1d(24, 32, kernel_size=5, stride=1, padding=1)
        self.conv6 = nn.Conv1d(32, 48, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(48, 60, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv1d(60, 64, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(64 * 273, latent_dim * 2)


    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv6(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=3)
        x = F.relu((self.conv7(x)))
        x = F.relu((self.conv8(x)))
        x = x.view(x.size(0), -1)
        mean, logvar = torch.chunk(self.fc(x), 2, dim=1)
        return mean, logvar

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
