import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

#Helper functions
def D(data):
    '''
    :param data: The data for which the distance matrix is supposed to be calculated
    :return: The finished distance matrix
    '''
    return distance_matrix(data,data)


def invert_diagonal(diag_matrix):
    '''
    :param diag_matrix: The diagonal matrix that is supposed to be inverted
    :return: the inverted diagonal matrix
    '''
    ret = np.zeros(diag_matrix.shape)
    for i in range(diag_matrix.shape[0]):
        ret[i,i] = 1/diag_matrix[i,i]
    return ret

def invert_root_diagonal(diag_matrix):
    '''
    :param diag_matrix: The diagonal matrix that is supposed to be inverted and taken the squareroot of
    :return: the inverted diagonal matrix
    '''
    ret = np.zeros(diag_matrix.shape)
    for i in range(diag_matrix.shape[0]):
        ret[i,i] = 1/np.sqrt(diag_matrix[i,i])
    return ret

#Diffusion map function
def diffusion_map(dataset, amount_of_vectors = 5):
    '''
    The diffusion Map algorithm
    :param dataset: The data for which the diffusion map algorithm is to be calculated
    :param amount_of_vectors: How many vectors are supposed to be returned (Largest ones, ascending order)
    :return: eigenvalues,eigenvectors in ascending order
    '''
    #First get the Size of the dataset
    N = len(dataset)
    #Create the distance matrix
    D_matrix = D(dataset)
    #Calculate epsilon (5% of the largest distance)
    epsilon = 0.05 * D_matrix.max()
    #Calculate the kernel as described in the exercise sheet
    #Note: Very numerically sensitive
    W = np.exp(-(np.square(D_matrix)/epsilon))
    #Create the diagonal matrix P for normalization
    P  = np.zeros(W.shape)
    for i in range(N):
        for j in range(N):
            P[i,i] = P[i,i] + W[i,j]
    #Alternatively create it like this_
    #P = np.diag(np.sum(W,axis=0))
    #But this gives different results? (Numerical instability probably)
    
    #Calculate the inverse of P
    #Note, because P is a diagonal matrix, the inverse is just a matrix where each diagonal element is 1 divided by the original element
    P_inv = invert_diagonal(P)
    #Alternatively can be calculated like this, but this gives different results? (Numerical instability probably)
    #P_inv = LA.inv(P)
    
    #Calculate K by normalizing W with P_inv
    K = P_inv @ W @ P_inv
    
    #Calculate Q the same way P was calculated
    Q = np.diag(np.sum(K,axis=0))
    #Q^-1/2 is the same as taking the square root of each diagonal element and calculating 1 divided by that value
    #This is again very numerically instable and can give varying results based on the way how one calculates it
    Q_rootinv = LA.inv(np.sqrt(Q))
    
    #Calculate T_hat (Here I just use T) by normalizing K with Q_rootinv
    T = Q_rootinv @ K @ Q_rootinv
    
    #Calculate the eigenvalues and eigenvectors of T, sorted in ascending order(!) 
    eigenvalues,eigenvectors = eigh(T,subset_by_index = [N-amount_of_vectors,N-1])
    
    #Calculate the final eigenvalues and eigenvectors as described in the exercise sheet
    lambda_l_squared = np.sqrt(np.power(eigenvalues,1/epsilon))
    theta_l = Q_rootinv @ eigenvectors
    
    return lambda_l_squared,theta_l