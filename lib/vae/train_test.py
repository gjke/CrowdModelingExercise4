import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epoch, model, data_loader, optimizer, loss_function, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, sigma = model(data)
        loss = loss_function(data, recon_batch, mu, sigma)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        '''
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data)))
        '''

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))


def test(model, data_loader, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (data, _) in data_loader:
            data = data.to(device)
            recon_batch, mu, sigma = model(data)
            test_loss += loss_function(data, recon_batch, mu, sigma).item()

    return test_loss / len(data_loader.dataset)
