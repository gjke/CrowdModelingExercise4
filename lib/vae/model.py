import torch
import torch.nn as nn
import torch.nn.functional as F

fn_dict = {
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


class VAE(nn.Module):
    def __init__(self, x_dim, enc_hid_dims, z_dim, dec_hid_dims, dec_norm_fn='sigmoid'):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        # encoder layers
        self.encoder = nn.ModuleList(
            [
                nn.Linear(x_dim, enc_hid_dims[0]),
                nn.ReLU(),

            ]
        )
        for i in range(1, len(enc_hid_dims)):
            self.encoder.extend([
                nn.Linear(enc_hid_dims[i-1], enc_hid_dims[i]),
                nn.ReLU(),
            ])

        self.eout1 = nn.Linear(enc_hid_dims[-1], z_dim)
        self.eout2 = nn.Linear(enc_hid_dims[-1], z_dim)

        # decoder layers
        self.decoder = nn.ModuleList(
            [
                nn.Linear(z_dim, dec_hid_dims[0]),
                nn.ReLU(),
            ]
        )
        for i in range(1, len(dec_hid_dims)):
            self.decoder.extend([
                nn.Linear(dec_hid_dims[i-1], enc_hid_dims[i]),
                nn.ReLU(),
            ])
        self.dout = nn.Linear(dec_hid_dims[-1], x_dim)
        self.dout_norm = fn_dict.get(dec_norm_fn, nn.Sigmoid)()

    def encode(self, x):
        y = x
        for i in range(len(self.encoder)):
            y = self.encoder[i](y)
        # mu, sigma of p(z|x) = N(z|mu, sigma)
        return self.eout1(y), self.eout2(y)

    def sample(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        eps = torch.randn_like(sigma)
        return mu + eps*sigma  # return z sample

    def decode(self, z):
        y = z
        for i in range(len(self.decoder)):
            y = self.decoder[i](y)
        return self.dout_norm(self.dout(y))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.x_dim))
        z = self.sample(mu, log_var)
        return self.decode(z), mu, log_var


def kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


def loss_function(x, x_recon, mu, log_var):
    return (
        ((x.view(-1, x_recon.shape[1]) - x_recon)**2).sum()   # ~ -log p(x|z)
        + kl_divergence(mu, log_var)
    )
