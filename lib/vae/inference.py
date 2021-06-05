import torch
from torch.distributions.normal import Normal


def encode_dataset(model, data_loader):
    z_batches = []
    target_batches = []
    for data, target in data_loader:
        z_batches.append(encode_batch(model, data))
        target_batches.append(target)
    return torch.cat(z_batches).numpy(), torch.cat(target_batches).numpy()


def encode_batch(model, batch):
    """
    Embeds the input into the latent space

    Parameters:
        model (VAE): VAE model
        batch (pytorch.Tensor): input images of size (n, 1, 28, 28), 
            where n is the batch length

    Returns:
        pytorch.Tensor: latent representasion of the input. Shape: (n, m), where n is
        the batch length and m is the dimention of the latent space  

    """
    model.eval()
    with torch.no_grad():
        mu, log_var = model.encode(batch.view(-1, model.x_dim))
        sigma = torch.exp(0.5*log_var)
        z = model.sample(mu, sigma)
    return z


def reconstruct(model, batch):
    """
    Embeds the input into the latent space

    Parameters:
        model (VAE): VAE model
        batch (pytorch.Tensor): input images of size (n, 1, 28, 28), 
            where n is the batch length

    Returns:
        pytorch.Tensor: reconstructed input. Shape: (n, 1, 28, 28), where n is
        the batch length  

    """
    model.eval()
    with torch.no_grad():
        reconstructed_x, _, _ = model(batch)

    return reconstructed_x.view(-1, 1, 28, 28)


def generate(model, n):
    """
    samples n from Normal(0, I) and decodes them using model

    Parameters:
        model (VAE): VAE model
        n (int): number of samples to generate

    Returns:
        pytorch.Tensor: reconstructed input. Shape: (n, 1, 28, 28)

    """

    prior_samples = Normal(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0])
    ).sample((n,))
    with torch.no_grad():
        generated = model.decode(prior_samples)

    return generated.view(-1, 1, 28, 28)
