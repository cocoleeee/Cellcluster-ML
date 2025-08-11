from model.utils import euclidean_dist
import torch

# M1 loss
def M1_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    #print('Reconstruction {}'.format(loss_rcn))
    return loss_rcn

# M2 loss
# def M2_loss(encoded, y, Lambda):
#     L1 = distance_L1(encoded, y)
#     L2 = mindistance_L2(encoded, y)
#     Loss = L1 + Lambda *L2 #超参数调整

#     return Loss, L1, L2


def contrastive_guidance(encoded, y, margin=1.0):
    """
    encoded: (B, D) tensor
    y: (B,) 0/1 标签，1 表示用户选中细胞
    margin: 负对之间的最小距离
    """
    device = encoded.device
    sel_mask = (y == 1)
    non_mask = (y == 0)
    z_sel = encoded[sel_mask]     # (Ns, D)
    z_non = encoded[non_mask]     # (Nn, D)

    # 1) 正对：拉近所有选中细胞两两距离（可选，如果 Ns 太大可以随机采样）
    if z_sel.shape[0] > 1:
        pdist_pos = euclidean_dist(z_sel, z_sel)            # (Ns, Ns)
        # 只取上三角
        pos_loss = pdist_pos.triu(diagonal=1).mean()
    else:
        pos_loss = torch.tensor(0., device=device)

    # 2) 负对：拉远所有选中 vs 非选中细胞对，margin-hinge
    if z_sel.shape[0] > 0 and z_non.shape[0] > 0:
        pdist_neg = euclidean_dist(z_sel, z_non)            # (Ns, Nn)
        # hinge: max(0, margin - dist)
        neg_loss = torch.relu(margin - pdist_neg).mean()
    else:
        neg_loss = torch.tensor(0., device=device)

    return pos_loss + neg_loss

def M2_loss(encoded, y, Lambda, gamma=0.1, margin=0.1):
    # 原来的原型损失
    L1 = distance_L1(encoded, y)
    L2 = mindistance_L2(encoded, y)
    proto_loss = L1 + Lambda * L2

    # 新加的样本级对比约束
    Lg = contrastive_guidance(encoded, y, margin)

    # 最终总损失
    Loss = proto_loss + gamma * Lg
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

