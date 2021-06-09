import numpy as np
from scipy import linalg


def center_data(data):
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    return data_centered


def pca(pc, data):
    """
    Principle Component Analysis 

    Parameters:
        pc(int): number of principle components
        data(numpy.array): dataset to be analysed

    Returns:
        (Projected point, Reconstructed Data, Energy of each component, V)
    """
    U, S, V = linalg.svd(data)

    U = U[:, 0:pc]

    S_matrix = np.zeros((U.shape[1], V.shape[0]))
    for i in range(min(S_matrix.shape[0], S_matrix.shape[1])):
        S_matrix[i, i] = S[i]

    projected_points = data.dot(V.T)
    variance = np.var(projected_points, axis=0)
    energy = variance / np.sum(variance)

    reconstructed_data = np.dot(U, np.dot(S_matrix, V))

    return projected_points, reconstructed_data, energy, V
