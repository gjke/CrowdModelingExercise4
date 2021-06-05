import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim, enc_hid_dim_1, enc_hid_dim_2, z_dim, dec_hid_dim_1, dec_hid_dim_2):
        super(VAE, self).__init__()

        self.x_dim = x_dim

        # encoder layers
        self.ehl1 = nn.Linear(x_dim, enc_hid_dim_1)
        self.ehl2 = nn.Linear(enc_hid_dim_1, enc_hid_dim_2)
        self.eout1 = nn.Linear(enc_hid_dim_2, z_dim)
        self.eout2 = nn.Linear(enc_hid_dim_2, z_dim)

        # decoder layers
        self.dhl1 = nn.Linear(z_dim, dec_hid_dim_1)
        self.dhl2 = nn.Linear(dec_hid_dim_1, dec_hid_dim_2)
        self.dout = nn.Linear(dec_hid_dim_2, x_dim)
        self.dout_norm = nn.Sigmoid()

    def encode(self, x):
        h = F.relu(self.ehl1(x))
        h = F.relu(self.ehl2(h))
        # mu, sigma of p(z|x) = N(z|mu, sigma)
        return self.eout1(h), self.eout2(h)

    def sample(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)  # return z sample

    def decode(self, z):
        h = F.relu(self.dhl1(z))
        h = F.relu(self.dhl2(h))
        return self.dout_norm(self.dout(h))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.x_dim))
        sigma = torch.exp(0.5*log_var)
        z = self.sample(mu, torch.exp(0.5*log_var))
        x_reconstr = self.decode(z)
        return x_reconstr, mu, sigma


def kl_divergence(mu, sigma):
    return 0.5 * (sigma**2 + mu**2 - torch.log(sigma) - 1).sum()


def loss_function(x, x_recon, mu, sigma):
    return (
        ((x.view(-1, 784) - x_recon)**2).sum() +  # ~ -log p(x|z)
        kl_divergence(mu, sigma)
    )
