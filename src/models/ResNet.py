import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Encoder based on ResNet-18
class Encoder(nn.Module):
    def __init__(self, latent_dim, input_length):
        super(Encoder, self).__init__()
        # Load ResNet-18 and modify it for 1D inputs
        self.resnet = models.resnet18(pretrained=False)

        # Modify the first convolutional layer to accept 1D input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.resnet.maxpool = nn.Identity()  # Remove maxpool, as it might reduce the input too early

        # Calculate the output dimension after the global average pooling
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Modify the final fully connected layer to output the desired latent dimension
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2 * latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = x.unsqueeze(-1)  # Add a spatial dimension to make it 2D
        x = self.resnet(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), -1)
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
        z = self.fc3(z)
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
