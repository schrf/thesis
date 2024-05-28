import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_one_size, hidden_two_size, z_size, dropout):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_one_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_one_size, hidden_two_size),
            nn.ReLU()
        )

        # latent space
        self.fc_mu = nn.Linear(hidden_two_size, z_size)
        self.fc_sigma = nn.Linear(hidden_two_size, z_size)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden_two_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_two_size, hidden_one_size),
            nn.ReLU(),
            nn.Linear(hidden_one_size, input_size)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + std * eps

    def forward(self, x):
        x = self.encoder(x)

        mu, sigma = self.fc_mu(x), self.fc_sigma(x)

        z = self.reparameterize(mu, sigma)

        x_recon = self.decoder(z)

        return x_recon, mu, sigma


class AE(nn.Module):
    def __init__(self, input_size, hidden_one_size, hidden_two_size, z_size):
        super().__init__()

        # encoding
        self.fc1 = nn.Linear(input_size, hidden_one_size)
        self.fc2 = nn.Linear(hidden_one_size, hidden_two_size)
        self.fc3 = nn.Linear(hidden_two_size, z_size)

        # decoding
        self.fc4 = nn.Linear(z_size, hidden_two_size)
        self.fc5 = nn.Linear(hidden_two_size, hidden_one_size)
        self.fc6 = nn.Linear(hidden_one_size, input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
