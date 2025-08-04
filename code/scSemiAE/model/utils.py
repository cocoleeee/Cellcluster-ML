import torch
import numpy as np
from torch.utils.data import DataLoader

def init_data_loaders(labeled_data, pretrain_data, pretrain_batch):
    '''
    Initialize loaders for pretraing and training (labeled datasets). 
    '''
    pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_data, shuffle=True,
                                                batch_size=pretrain_batch if pretrain_batch!=None else len(pretrain_data.x))
    train_loader = DataLoader(dataset=labeled_data, shuffle=True, batch_size=100)

    return train_loader, pretrain_loader
           
           
def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)

    if x.ndim == 1:
        d = n
        n = 1
        x = x.unsqueeze(0)
    else:
        d = x.size(1)

    if y.ndim == 1 and d == m:
        m = 1
        y = y.unsqueeze(0)
    elif d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def euclidean_distance(x):
    """
    Compute euclidean distance of a tensor
    """
    n = x.shape[0]
    matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            matrix[i][j] = np.square(x[i]-x[j]).sum()

    return matrix