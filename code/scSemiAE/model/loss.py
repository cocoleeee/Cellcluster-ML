from model.utils import euclidean_dist
import torch
import torch.nn.functional as F


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


# def user_pull_loss(encoded, y):
#     """
#     encoded: (B, D)
#     y:       (B,) 0/1，1 表示用户选择
#     """
#     mask = (y == 1)
#     if mask.sum() < 2:
#         return encoded.new_tensor(0., device=encoded.device)
#     z = F.normalize(encoded, dim=1)
#     pu = F.normalize(z[mask].mean(0, keepdim=True), dim=1)  # 用户原型
#     return (1.0 - (z[mask] * pu).sum(1)).mean()             # 1 - cos

# # --- 你现有的对比约束（保持不变即可，建议 margin 和负样本数稍调大些） ---
# def contrastive_guidance(encoded, y, margin=0.25, neg_per_pos=64, eps=0.05):
#     device = encoded.device
#     idx_pos = (y==1).nonzero(as_tuple=False).squeeze()
#     idx_neg = (y==0).nonzero(as_tuple=False).squeeze()
#     if idx_pos.numel() < 2 or idx_neg.numel() < 1:
#         return torch.tensor(0., device=device)

#     z_pos = encoded[idx_pos]           # (Ns, D)
#     z_neg_all = encoded[idx_neg]       # (Nn, D)

#     # 正对：选中的两两拉近（上三角平均）
#     pd_pos = euclidean_dist(z_pos, z_pos)
#     pos_loss = pd_pos.triu(1).mean()

#     # 负对：半硬负
#     neg_loss_list = []
#     for i in range(z_pos.size(0)):
#         perm = torch.randperm(z_neg_all.size(0), device=device)[:neg_per_pos]
#         z_neg = z_neg_all[perm]
#         dist_neg = euclidean_dist(z_pos[i:i+1], z_neg).view(-1)
#         hard = dist_neg < (margin + eps)
#         if hard.any():
#             neg_loss_list.append(torch.relu(margin - dist_neg[hard]).mean())
#     neg_loss = torch.stack(neg_loss_list).mean() if neg_loss_list else torch.tensor(0., device=device)

#     return pos_loss + neg_loss

# import torch
# import torch.nn.functional as F

# def center_cosine_loss_weighted(z, y, w_user=2.0):
#     """
#     余弦中心损失：让每个样本靠近其所属类原型；对 y==1 给更大权重（强调先验）。
#     L1 := mean_i w(y_i) * (1 - cos(z_i, p_{y_i}))
#     """
#     # 归一化
#     z = F.normalize(z, dim=1)
#     # 重编号（不改原 y）
#     with torch.no_grad():
#         uniq = torch.unique(y.detach().cpu(), sorted=True)
#         mapping = {int(v): i for i, v in enumerate(uniq.tolist())}
#     y_ = y.clone()
#     for v,i in mapping.items(): y_[y==v] = i
#     C = len(uniq)
#     # 原型
#     protos = []
#     for c in range(C):
#         idx = (y_==c).nonzero(as_tuple=False).squeeze(-1)
#         p = z[idx].mean(0) if idx.numel()>0 else torch.zeros(z.size(1), device=z.device)
#         protos.append(p)
#     P = F.normalize(torch.stack(protos, dim=0), dim=1)  # (C,D)
#     sims = (z * P[y_]).sum(1)                            # cos(z_i, p_{y_i})
#     w = torch.ones_like(sims)
#     # 给“用户类”（假定原 y==1）更大权重；若你的用户标签不是 1，请改成相应值
#     w = torch.where((y==1), w_user*w, w)
#     return (w * (1.0 - sims)).mean()

# def user_inter_margin_loss(z, y, margin=0.25):
#     """
#     只对“用户原型 vs 其它原型”施加角度 margin：
#     L2 := mean_j max(0, margin - (1 - cos(p_user, p_j)))  for j!=user
#     更聚焦“把用户和他类拉开”，不会到处推开所有类。
#     """
#     z = F.normalize(z, dim=1)
#     # 重编号
#     with torch.no_grad():
#         uniq = torch.unique(y.detach().cpu(), sorted=True)
#         mapping = {int(v): i for i, v in enumerate(uniq.tolist())}
#     y_ = y.clone()
#     for v,i in mapping.items(): y_[y==v] = i
#     C = len(uniq)
#     # 原型
#     protos = []
#     for c in range(C):
#         idx = (y_==c).nonzero(as_tuple=False).squeeze(-1)
#         p = z[idx].mean(0) if idx.numel()>0 else torch.zeros(z.size(1), device=z.device)
#         protos.append(p)
#     P = F.normalize(torch.stack(protos, dim=0), dim=1)  # (C,D)

#     # 找出“用户类”的索引（原 y==1 对应到重编号后的 id）
#     # 若你的用户标签不是 1，请改这里的条件
#     user_vals = [mapping.get(1, None)]
#     if user_vals[0] is None:  # 没有用户类就返回 0
#         return z.new_tensor(0.)
#     u = user_vals[0]
#     if C<=1: return z.new_tensor(0.)

#     cos = (P[u:u+1] @ P.t()).squeeze(0)  # (C,)
#     d = 1.0 - cos
#     mask = torch.ones(C, dtype=torch.bool, device=z.device); mask[u]=False
#     d_others = d[mask]
#     return F.relu(margin - d_others).mean()






# def M2_loss(encoded, y, Lambda, gamma=0.3, margin=0.25, user_w=2):
#     """
#     encoded: (B,D)
#     y:       (B,) 0/1，其中 1=用户选择
#     Lambda:  L2 的权重
#     gamma:   contrastive_guidance 的权重
#     user_w:  用户拉拢项权重（0.4~0.8 常用）
#     """
#     # 1) 原型损失（注意传 y.clone() 避免被 distance_* 原地修改）
#     L1 = distance_L1(encoded, y.clone())
#     L2 = mindistance_L2(encoded, y.clone())
#     proto_loss = L1 + Lambda * L2

#     # 2) 对比约束（建议 margin=0.25，neg_per_pos=64）
#     Lg = contrastive_guidance(encoded, y, margin=margin, neg_per_pos=64, eps=0.05)

#     # 3) 用户拉拢（角度中心）
#     L_user = user_pull_loss(encoded, y)

#     Loss = proto_loss + gamma * Lg + user_w * L_user
#     return Loss, L1, L2


# def M2_loss(encoded, y, Lambda,      # Lambda 占位保留（不再使用欧氏 L2）
#             gamma=0.1, margin=0.25, user_w=0.6,
#             w_user_center=2.0):
#     """
#     encoded: (B,D)
#     y:       (B,) 0/1，其中 1=用户选择
#     Lambda:  占位，保持接口兼容
#     gamma:   对比项权重
#     margin:  用户原型 vs 他类原型的角度 margin
#     user_w:  用户拉拢项权重
#     w_user_center: 类内中心损失里对 y==1 的样本权重（2.0~3.0 常用）
#     """
#     # L1：类内紧致（角度版；对 y==1 加权）
#     L1 = center_cosine_loss_weighted(encoded, y, w_user=w_user_center)
#     # L2：只盯“用户 vs 他类”的 margin 分离
#     L2 = user_inter_margin_loss(encoded, y, margin=margin)
#     # 样本级对比（沿用你现成的函数；建议 margin=0.25, neg_per_pos=64）
#     Lg = contrastive_guidance(encoded, y, margin=margin, neg_per_pos=64, eps=0.05)
#     # 用户拉拢
#     L_user = user_pull_loss(encoded, y)

#     Loss = L1 + L2 + gamma*Lg + user_w*L_user
#     return Loss, L1, L2




# # -----------------------------
# # 工具：安全重编号 + 原型计算（不改原 target）
# # -----------------------------
# def _relabel(target: torch.Tensor):
#     """返回 relabeled(0..C-1) 及每类索引 list，不改原 target。"""
#     with torch.no_grad():
#         uniq = torch.unique(target.detach().cpu(), sorted=True)
#         mapping = {int(v): i for i, v in enumerate(uniq.tolist())}
#         relabeled = target.clone()
#         for v, i in mapping.items():
#             relabeled[target == v] = i
#         class_idxs = [ (relabeled == i).nonzero(as_tuple=False).squeeze(-1)
#                        for i in range(len(uniq)) ]
#     return relabeled.long(), class_idxs, len(uniq)

# def _class_prototypes(z: torch.Tensor, class_idxs, l2_normalize=True, detach_proto=True):
#     """按类均值计算原型；可选择 L2 归一化与 stop-grad。"""
#     protos = []
#     for idx in class_idxs:
#         if idx.numel() == 0:
#             protos.append(torch.zeros(z.size(1), device=z.device))
#         else:
#             p = z[idx].mean(0)
#             protos.append(p)
#     P = torch.stack(protos, dim=0)  # (C, D)
#     if l2_normalize:
#         P = F.normalize(P, dim=1)
#     if detach_proto:
#         P = P.detach()
#     return P

# # -----------------------------
# # 2) 类内“中心/角度”损失：更稳的聚内紧致
# #     L_intra = mean_i (1 - cos(z_i, p_{y_i}))
# # -----------------------------
# def center_cosine_loss(z, target):
#     y, class_idxs, C = _relabel(target)
#     z_n = F.normalize(z, dim=1)
#     P = _class_prototypes(z_n, class_idxs, l2_normalize=False, detach_proto=True)  # P 已归一
#     P = F.normalize(P, dim=1)
#     # 取每个样本对应原型的 cos，相当于 1 - 相似度
#     sims = torch.sum(z_n * P[y], dim=1)  # (N,)
#     return (1.0 - sims).mean()

# # -----------------------------
# # 3) 原型间 margin 分离（角度版）
# #     对所有 i!=j：max(0, m - (1 - cos(p_i, p_j))) = max(0, m - d_ij)
# # -----------------------------
# def inter_proto_margin_loss(z, target, margin=0.2):
#     y, class_idxs, C = _relabel(target)
#     if C <= 1:
#         return z.new_tensor(0.)
#     P = _class_prototypes(z, class_idxs, l2_normalize=True, detach_proto=True)  # (C, D), L2 归一
#     # 角度距离 d = 1 - cos
#     cos = P @ P.t()                  # (C, C)
#     d = 1.0 - cos
#     mask = ~torch.eye(C, dtype=torch.bool, device=z.device)
#     d_off = d[mask]
#     return F.relu(margin - d_off).mean()

# # -----------------------------
# # 4) PrototypeNCE（正=用户原型/类原型；负=批内/缓存） 
# #    用于“用户选择细胞” y==1 的个性化对比
# # -----------------------------
# def prototype_nce_loss(z, y, p_user=None, neg_bank=None, K=128, tau=0.2):
#     """
#     z: (B,D) encoder 输出
#     y: (B,) 二分类标签，1 表示用户选择或相似类
#     p_user: (D,) 或 (1,D) 用户原型；若 None，则用 batch 内 y==1 的均值
#     neg_bank: (M,D) 负样本缓存（已 L2 归一）；可为 None
#     """
#     device = z.device
#     idx_pos = (y == 1).nonzero(as_tuple=False).squeeze(-1)
#     if idx_pos.numel() == 0:
#         return z.new_tensor(0.)

#     z_n = F.normalize(z, dim=1)
#     if p_user is None:
#         pu = F.normalize(z[idx_pos].mean(0, keepdim=True), dim=1)  # (1,D)
#     else:
#         pu = p_user.view(1, -1)
#         pu = F.normalize(pu, dim=1)                                # (1,D)

#     # 负样本：优先从 neg_bank 取；否则从 y==0 的 batch 中取
#     if neg_bank is not None and neg_bank.numel() > 0:
#         M = neg_bank.size(0)
#         pick = torch.randint(0, M, (idx_pos.numel(), K), device=device)
#         z_negs = neg_bank[pick]                                   # (Bpos,K,D)
#     else:
#         idx_neg = (y == 0).nonzero(as_tuple=False).squeeze(-1)
#         if idx_neg.numel() == 0:
#             return z.new_tensor(0.)
#         perm = torch.randint(0, idx_neg.numel(), (idx_pos.numel(), K), device=device)
#         z_negs = z_n[idx_neg[perm]]                               # (Bpos,K,D)

#     anchors = z_n[idx_pos]                                        # (Bpos,D)
#     pu_expand = pu.expand_as(anchors)                             # (Bpos,D)

#     # logits = [pos, negs] / tau
#     pos = torch.sum(anchors * pu_expand, dim=1, keepdim=True) / tau                  # (Bpos,1)
#     neg = torch.einsum('bd,bkd->bk', anchors, z_negs) / tau                          # (Bpos,K)
#     logits = torch.cat([pos, neg], dim=1)                                            # (Bpos,1+K)
#     labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
#     return F.cross_entropy(logits, labels)

# # -----------------------------
# # 5) 组合成 M2：类内紧致 + 原型间分离 + PrototypeNCE
# #    （保留你的 Lambda 槽位以兼容旧接口）
# # -----------------------------

# def user_pull_loss(encoded, y):
#     """
#     把 y==1 的样本在 latent 里往它们自己的均值方向拉（角度/余弦形式）。
#     encoded: (B, D)
#     y:       (B,)  0/1，1 表示用户选择的细胞
#     """
#     mask = (y == 1)
#     if mask.sum() < 2:
#         return encoded.new_tensor(0., device=encoded.device)
#     z = F.normalize(encoded, dim=1)
#     pu = F.normalize(z[mask].mean(0, keepdim=True), dim=1)   # 用户原型
#     return (1.0 - (z[mask] * pu).sum(1)).mean()              # 1 - cos





# def M2_loss(encoded, y, Lambda=1.0,  # Lambda 保留以兼容旧代码
#             proto_w=0.5, inter_w=0.2, nce_w=0.3,user_w=0.6,
#             margin=0.2, tau=0.2, neg_bank=None, p_user=None):
#     """
#     encoded: (B,D)
#     y: (B,) 多类或二类；若是你的 0/1(用户/非用户)，同样适用
#     p_user: 若提供，则对 y==1 的个性化对比用该原型（(D,) 或 (1,D)）
#     neg_bank: 负样本缓存 (M,D)，建议已归一
#     """
#     # 1) 类内紧致（角度中心）
#     L_intra = center_cosine_loss(encoded, y)

#     # 2) 原型间分离（所有类之间的 margin hinge）
#     L_inter = inter_proto_margin_loss(encoded, y, margin=margin)

#     # 3) 个性化 PrototypeNCE（只作用于 y==1）
#     L_nce = prototype_nce_loss(encoded, (y == 1).long(), p_user=p_user, neg_bank=neg_bank, K=128, tau=tau)

#     L_user = user_pull_loss(encoded, y)
#     Loss = proto_w * L_intra + inter_w * L_inter + nce_w * L_nce+user_w * L_user
#     return Loss, L_nce, L_user





# def distance_L1(data, target):
#     uniq = torch.unique(target, sorted=True)
#     class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))
                            
#     for idx,v in enumerate(uniq):
#         target[target==v] = idx
    
#     prototypes = torch.stack([data[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
#     dists = euclidean_dist(data, prototypes)
#     loss = torch.stack([dists[idx_example, idx_proto].mean(0) for idx_proto,idx_example in enumerate(class_idxs)]).mean()
#     return loss

# def distance_L2(data, target):
#     uniq = torch.unique(target, sorted=True)
#     class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))

#     for idx,v in enumerate(uniq):
#         target[target==v] = idx
    
#     prototypes = torch.stack([data[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
#     dists = euclidean_dist(prototypes, prototypes)
#     nproto = prototypes.shape[0]
#     loss = - torch.sum(dists) / (nproto*nproto-nproto)

#     return loss


def distance_L1(data, target):
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))
    for idx,v in enumerate(uniq):
        target[target==v] = idx
    prototypes = torch.stack([data[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
    dists = euclidean_dist(data, prototypes)
    return torch.stack([dists[idx_example, idx_proto].mean(0) for idx_proto,idx_example in enumerate(class_idxs)]).mean()

def mindistance_L2(data, target):
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))
    for idx, v in enumerate(uniq):
        target[target == v] = idx
    prototypes = torch.stack([data[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
    dists = euclidean_dist(prototypes, prototypes)
    nproto = prototypes.shape[0]
    distances = torch.zeros(nproto, nproto-1, device=data.device)
    if dists.shape[0] == 1:
        min_dist = torch.min(dists, 1)[0]
    else:
        for i in range(nproto):
            distances[i] = torch.cat((dists[i][:i], dists[i][i+1:]), 0)
        min_dist = torch.min(distances, 1)[0]
    return - torch.min(min_dist)

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

