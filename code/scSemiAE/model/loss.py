from model.utils import euclidean_dist
import torch

# M1 loss
def M1_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    #print('Reconstruction {}'.format(loss_rcn))
    return loss_rcn

# M2 loss
def M2_loss(encoded, y, Lambda):
    L1 = distance_L1(encoded, y)
    L2 = mindistance_L2(encoded, y)
    Loss = L1 + Lambda *L2 #超参数调整

    return Loss, L1, L2

def distance_L1(data, target):
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))
                            
    for idx,v in enumerate(uniq):
        target[target==v] = idx
    
    prototypes = torch.stack([data[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
    dists = euclidean_dist(data, prototypes)
    loss = torch.stack([dists[idx_example, idx_proto].mean(0) for idx_proto,idx_example in enumerate(class_idxs)]).mean()
    return loss

def distance_L2(data, target):
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))

    for idx,v in enumerate(uniq):
        target[target==v] = idx
    
    prototypes = torch.stack([data[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
    dists = euclidean_dist(prototypes, prototypes)
    nproto = prototypes.shape[0]
    loss = - torch.sum(dists) / (nproto*nproto-nproto)

    return loss

def mindistance_L2(data, target):
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))

    for idx, v in enumerate(uniq):
        target[target == v] = idx

    prototypes = torch.stack([data[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
    dists = euclidean_dist(prototypes, prototypes)
    nproto = prototypes.shape[0]
    distances = torch.zeros(nproto, nproto-1)
    if dists.shape[0] == 1:
        min_dist = torch.min(dists, 1)[0]
    else:
        for i in range(nproto):
            distances[i] = torch.cat((dists[i][:i],dists[i][i+1:]), 0)
        min_dist = torch.min(distances, 1)[0]

    loss = - torch.min(min_dist)

    return loss

