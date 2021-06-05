from matplotlib import pyplot as plt


def plot_latent_representation(z, targets, title):
    _, ax = plt.subplots()
    scatter = ax.scatter(
        z[:, 0], z[:, 1], c=targets, label=targets)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.title(title)
    plt.show()


def plot_reconstructed(initial, reconstructed, title):
    n = 15
    f, axarr = plt.subplots(3, 10)
    for i in range(n):
        row = i // 5
        col = i % 5

        axarr[row, 2*col].imshow(initial[i][0],
                                 cmap='gray', interpolation='none')
        axarr[row, 2*col+1].imshow(reconstructed
                                   [i][0], cmap='gray', interpolation='none')
        axarr[row, 2*col].axes.xaxis.set_visible(False)
        axarr[row, 2*col].axes.yaxis.set_visible(False)
        axarr[row, 2*col+1].axes.xaxis.set_visible(False)
        axarr[row, 2*col+1].axes.yaxis.set_visible(False)
    axarr[0, 4].set_title(title)
    plt.show()


def plot_generated(generated, title):
    n = 15
    f, axarr = plt.subplots(3, 5)
    for i in range(n):
        row = i // 5
        col = i % 5

        axarr[row, col].imshow(generated[i][0],
                               cmap='gray', interpolation='none')
        axarr[row, col].axes.xaxis.set_visible(False)
        axarr[row, col].axes.yaxis.set_visible(False)
    axarr[0, 2].set_title(title)
    plt.show()


def plot_loss(loss_per_epoch, title):
    _, ax = plt.subplots()
    ax.plot(
        [epoch + 1 for epoch in range(len(loss_per_epoch))],
        loss_per_epoch
    )
    ax.set_title(title)
    plt.show()
