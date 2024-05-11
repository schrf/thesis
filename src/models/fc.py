import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_one_size, hidden_two_size, z_size):
        super().__init__()

        # encoder
        self.fc1 = nn.Linear(input_size, hidden_one_size)
        self.fc2 = nn.Linear(hidden_one_size, hidden_two_size)
        self.fc_mu = nn.Linear(hidden_two_size, z_size)
        self.fc_sigma = nn.Linear(hidden_two_size, z_size)

        # decoder
        self.fc3 = nn.Linear(z_size, hidden_two_size)
        self.fc4 = nn.Linear(hidden_two_size, hidden_one_size)
        self.fc5 = nn.Linear(hidden_one_size, input_size)


    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        return z


    def forward(self, x):
        mu, sigma = self.encode(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        x_recon = self.decode(z)
        return x_recon, mu, sigma
