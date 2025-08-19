from model.utils import euclidean_dist
import torch
import torch.nn.functional as F


# M1 loss
def M1_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    #print('Reconstruction {}'.format(loss_rcn))
    return loss_rcn

# # M2 loss
# def M2_loss(encoded, y, Lambda):
#     L1 = distance_L1(encoded, y)
#     L2 = mindistance_L2(encoded, y)
#     Loss = L1 + Lambda *L2 #超参数调整

#     return Loss, L1, L2


def user_pull_loss(encoded, y):
    """
    encoded: (B, D)
    y:       (B,) 0/1，1 表示用户选择
    """
    mask = (y == 1)
    if mask.sum() < 2:
        return encoded.new_tensor(0., device=encoded.device)
    z = F.normalize(encoded, dim=1)
    pu = F.normalize(z[mask].mean(0, keepdim=True), dim=1)  # 用户原型
    return (1.0 - (z[mask] * pu).sum(1)).mean()             # 1 - cos

# --- 你现有的对比约束（保持不变即可，建议 margin 和负样本数稍调大些） ---
def contrastive_guidance(encoded, y, margin=0.25, neg_per_pos=64, eps=0.05):
    device = encoded.device
    idx_pos = (y==1).nonzero(as_tuple=False).squeeze()
    idx_neg = (y==0).nonzero(as_tuple=False).squeeze()
    if idx_pos.numel() < 2 or idx_neg.numel() < 1:
        return torch.tensor(0., device=device)

    z_pos = encoded[idx_pos]           # (Ns, D)
    z_neg_all = encoded[idx_neg]       # (Nn, D)

    # 正对：选中的两两拉近（上三角平均）
    pd_pos = euclidean_dist(z_pos, z_pos)
    pos_loss = pd_pos.triu(1).mean()

    # 负对：半硬负
    neg_loss_list = []
    for i in range(z_pos.size(0)):
        perm = torch.randperm(z_neg_all.size(0), device=device)[:neg_per_pos]
        z_neg = z_neg_all[perm]
        dist_neg = euclidean_dist(z_pos[i:i+1], z_neg).view(-1)
        hard = dist_neg < (margin + eps)
        if hard.any():
            neg_loss_list.append(torch.relu(margin - dist_neg[hard]).mean())
    neg_loss = torch.stack(neg_loss_list).mean() if neg_loss_list else torch.tensor(0., device=device)

    return pos_loss + neg_loss

def user_center_margin_loss(encoded, y, user_center, m=0.3, k_neg=64):
    """
    把 y==1 的样本朝 user_center 拉近，并让 y==0 相对 center 的相似度
    不高于 y==1 的平均相似度减去 margin（角度/余弦）。
    """
    if user_center is None:
        return encoded.new_tensor(0., device=encoded.device)

    z = F.normalize(encoded, dim=1)
    c = F.normalize(user_center, dim=0)

    pos = z[y==1]
    neg = z[y==0]
    if pos.size(0) == 0:
        return encoded.new_tensor(0., device=encoded.device)

    # 拉拢：1 - cos
    L_pull = (1 - (pos @ c).clamp(-1, 1)).mean()

    # 推开：随机取 k 个负样本，相对正样本均值建立 margin
    if neg.size(0) > 0:
        k = min(k_neg, neg.size(0))
        pick = torch.randint(0, neg.size(0), (k,), device=encoded.device)
        s_pos = (pos @ c).mean()
        s_neg = (neg[pick] @ c)
        L_push = F.relu(s_neg - s_pos + m).mean()
    else:
        L_push = encoded.new_tensor(0., device=encoded.device)

    return L_pull + L_push

def M2_loss(encoded, y, Lambda,
            gamma=0.1, margin=0.3, user_w=1, center_w=0.8,
            user_center=None):
    """
    新增:
      center_w: 用户中心项的权重
      user_center: 训练循环里维护的 EMA 用户中心向量 (D,)
    其余保持原有极简配方：原型 + 对比 + 用户拉拢 + 用户中心margin
    """
    # 原型损失（注意 y.clone() 防止就地修改）
    L1 = distance_L1(encoded, y.clone())
    L2 = mindistance_L2(encoded, y.clone())
    proto_loss = L1 + Lambda * L2

    # 对比（建议 margin=0.25, neg_per_pos=64）
    Lg = contrastive_guidance(encoded, y, margin=margin, neg_per_pos=128, eps=0.1)

    # 用户拉拢
    L_user = user_pull_loss(encoded, y)

    # 用户中心（单向量）硬 margin
    L_center = user_center_margin_loss(encoded, y, user_center, m=margin, k_neg=64)

    Loss = proto_loss + gamma * Lg + user_w * L_user + center_w * L_center
    return Loss, L1, L2




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


# ===== Multi-Center user prototypes (EMA) =====
import torch
import torch.nn.functional as F

class MultiCenter:
    """
    维护 K 个用户中心（单位球面），用 EMA 更新；不参与反传（更稳）。
    """
    def __init__(self, D, K=3, beta=0.9, device="cpu"):
        self.K = int(K)
        self.beta = float(beta)
        self.device = device
        self.centers = torch.nn.Parameter(torch.randn(K, D, device=device), requires_grad=False)
        with torch.no_grad():
            self.centers.data = F.normalize(self.centers.data, dim=1)

    @torch.no_grad()
    def ema_update(self, z_pos: torch.Tensor):
        """用当前 batch 的 y==1 样本，对最近中心做一次 EMA。"""
        if z_pos is None or z_pos.numel() == 0:
            return
        z = F.normalize(z_pos, dim=1)                      # (Bpos, D)
        C = F.normalize(self.centers.data, dim=1)          # (K, D)
        sims = z @ C.t()                                   # (Bpos, K)
        assign = sims.argmax(dim=1)                        # (Bpos,)
        for k in range(self.K):
            idx = (assign == k).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                ck = F.normalize(z[idx].mean(0), dim=0)
                self.centers.data[k] = F.normalize(
                    self.beta * self.centers.data[k] + (1 - self.beta) * ck, dim=0
                )

def multicenter_user_loss(encoded, y, mc: MultiCenter, m=0.30, k_neg=128):
    """
    Pull：y==1 到“各自最近中心”的（1 - cos）；
    Push：y==0 对任一中心的最大 cos 不应超过 y==1 的平均“最近中心 cos”减去 margin。
    """
    if mc is None or (y == 1).sum() == 0:
        return encoded.new_tensor(0., device=encoded.device)

    z = F.normalize(encoded, dim=1)
    pos = z[y == 1]
    neg = z[y == 0]
    C = F.normalize(mc.centers.data.to(z.device), dim=1)   # (K, D)

    # Pull
    sims_pos = pos @ C.t()                                  # (Bpos, K)
    best = sims_pos.argmax(dim=1, keepdim=True)             # (Bpos,1)
    L_pull = (1.0 - sims_pos.gather(1, best)).mean()

    # Push
    if neg.numel() > 0:
        k = min(int(k_neg), neg.size(0))
        perm = torch.randint(0, neg.size(0), (k,), device=z.device)
        s_neg_max = (neg[perm] @ C.t()).max(dim=1).values   # (k,)
        s_pos_ref = sims_pos.max(dim=1).values.mean()       # 正样本最近中心的平均 cos
        L_push = F.relu(s_neg_max - s_pos_ref + m).mean()
    else:
        L_push = z.new_tensor(0., device=z.device)

    return L_pull + L_push

import torch, torch.nn.functional as F
from model.utils import euclidean_dist

# --- 安全重编号 ---
def _relabel_safe(y):
    with torch.no_grad():
        uniq = torch.unique(y.detach().cpu(), sorted=True)
        mapping = {int(v): i for i, v in enumerate(uniq.tolist())}
    y_ = y.clone()
    for v,i in mapping.items(): y_[y==v] = i
    idxs = [(y_==i).nonzero(as_tuple=False).squeeze(-1) for i in range(len(uniq))]
    return y_.long(), idxs

# --- L1（角度版）对 y==1 加权 ---
def center_cosine_loss_weighted(z, y, w_user=2.0):
    z = F.normalize(z, dim=1)
    y_, idxs = _relabel_safe(y)
    P = []
    for idx in idxs:
        P.append(z[idx].mean(0) if idx.numel()>0 else torch.zeros(z.size(1), device=z.device))
    P = F.normalize(torch.stack(P,0), dim=1)
    sims = (z * P[y_]).sum(1)                 # cos(z, p_class)
    w = torch.where((y==1), torch.tensor(w_user, device=z.device), torch.tensor(1.0, device=z.device))
    return (w * (1.0 - sims)).mean()

# --- 仅“用户 vs 其它”的中心间 margin（角度距离） ---
def user_vs_others_margin(z, y, margin=0.30):
    z = F.normalize(z, dim=1)
    y_, idxs = _relabel_safe(y)
    P = []
    for idx in idxs:
        P.append(z[idx].mean(0) if idx.numel()>0 else torch.zeros(z.size(1), device=z.device))
    P = F.normalize(torch.stack(P,0), dim=1)
    # 找用户类在重编号后的 id（原标签 1 -> new id）
    # 若你的用户标签不是 1，这里按实际值改
    uniq = torch.unique(y.detach().cpu(), sorted=True).tolist()
    if 1 not in uniq or len(uniq)<2: 
        return z.new_tensor(0.)
    u = uniq.index(1)
    cos = (P[u:u+1] @ P.t()).squeeze(0)     # (C,)
    d = 1.0 - cos
    mask = torch.ones_like(d, dtype=torch.bool); mask[u]=False
    return F.relu(margin - d[mask]).mean()

# --- 多中心（用户） ---
class MultiCenter:
    def __init__(self, D, K=3, beta=0.9, device="cpu"):
        self.K, self.beta = K, beta
        self.centers = torch.nn.Parameter(torch.randn(K, D, device=device), requires_grad=False)
        with torch.no_grad(): self.centers.data = F.normalize(self.centers.data, dim=1)
    @torch.no_grad()
    def ema_update(self, z_pos):
        if z_pos is None or z_pos.numel()==0: return
        z = F.normalize(z_pos, dim=1); C = F.normalize(self.centers.data, dim=1)
        a = (z @ C.t()).argmax(1)  # 最近中心
        for k in range(self.K):
            idx = (a==k).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel()>0:
                ck = F.normalize(z[idx].mean(0), dim=0)
                self.centers.data[k] = F.normalize(self.beta*self.centers.data[k] + (1-self.beta)*ck, dim=0)

def multicenter_user_loss(z, y, mc: MultiCenter, m=0.30, k_neg=128):
    if mc is None or (y==1).sum()==0: 
        return z.new_tensor(0., device=z.device)
    z = F.normalize(z, dim=1)
    pos, neg = z[y==1], z[y==0]
    C = F.normalize(mc.centers.data.to(z.device), dim=1)
    sims_pos = pos @ C.t()
    L_pull = (1.0 - sims_pos.max(1).values).mean()
    if neg.numel()>0:
        k = min(k_neg, neg.size(0))
        pick = torch.randint(0, neg.size(0), (k,), device=z.device)
        s_neg = (neg[pick] @ C.t()).max(1).values
        s_pos = sims_pos.max(1).values.mean()
        L_push = F.relu(s_neg - s_pos + m).mean()
    else:
        L_push = z.new_tensor(0.)
    return L_pull + L_push

# --- 总损失（精简版） ---
def M2_personal(encoded, y, w_user=2.0, margin=0.30, w_mc=0.8, mc=None):
    L1 = center_cosine_loss_weighted(encoded, y, w_user=w_user)
    L2 = user_vs_others_margin(encoded, y, margin=margin)
    Lmc = multicenter_user_loss(encoded, y, mc, m=margin, k_neg=128)
    return L1 + L2 + w_mc*Lmc

import torch
import torch.nn.functional as F

# ---------- 工具 ----------
def _l2n(z):  # L2 normalize
    return F.normalize(z, dim=1)

# 用户内方差（越小越紧）
def user_variance_loss(z, y, eps=1e-6):
    mask = (y == 1)
    if mask.sum() < 2:
        return z.new_tensor(0.)
    z = _l2n(z)
    mu = z[mask].mean(0, keepdim=True)          # (1,D)
    var = ( (z[mask] - mu)**2 ).sum(1)          # 每个样本到中心的平方和
    return var.mean()

# 二类监督对比：用户为同类，其余为异类（SupCon 简化）
def supcon_user_vs_rest(z, y, tau=0.2, margin=0.2):
    """
    z: (B,D), 任意 batch
    y: (B,), 1=用户，0=非用户
    tau: 温度
    margin: 通过 logit 偏置实现软 margin（把负对再往外推一点）
    """
    pos = (y == 1)
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return z.new_tensor(0.)

    z = _l2n(z)
    z_pos = z[pos]                           # (Np,D)
    z_all = z                                # (B,D)

    # 余弦相似度矩阵（pos 与 all）
    sim = (z_pos @ z_all.t()) / tau          # (Np,B)

    # 构造标签：同类=pos 自身及其它用户；异类=所有非用户
    mask_pospos = torch.zeros_like(sim, dtype=torch.bool)
    mask_pospos[:, pos] = True               # 同类列

    mask_posneg = torch.zeros_like(sim, dtype=torch.bool)
    mask_posneg[:, neg] = True               # 异类列

    # logits: [正, 负]；给负对加一个 -margin 的偏置（等价于再推远一点）
    pos_logits = sim.masked_fill(~mask_pospos, float('-inf'))  # 只保留正对
    neg_logits = sim.masked_fill(~mask_posneg, float('-inf')) - (margin / tau)

    # logsumexp 计算分母；分子是正对 logsumexp
    lse_pos = torch.logsumexp(pos_logits, dim=1)        # (Np,)
    lse_all = torch.logsumexp(torch.stack([pos_logits, neg_logits], dim=0).logsumexp(dim=0), dim=1)

    # SupCon: -log(正/正+负)
    loss = -(lse_pos - lse_all).mean()
    return loss

# 单位球正则（防止范数漂移）
def sphere_reg(z, target_norm=1.0):
    return ((z.norm(p=2, dim=1) - target_norm)**2).mean()

# ----------- 主损失（精简版）-----------
def M2_loss2(encoded, y,
            Lambda=0.0,                # 兼容占位，不再使用原 L2 原型互推
            w_var=1.0,                 # 用户内方差权重
            w_con=1.0,                 # 用户 vs 其余 SupCon 权重
            w_sph=0.05,                # 单位球正则
            tau=0.2, margin=0.25):     # SupCon 超参
    """
    encoded: (B,D)
    y:       (B,) 0/1，1=用户
    """
    L_var = user_variance_loss(encoded, y)                   # 用户内紧致
    L_con = supcon_user_vs_rest(encoded, y, tau=tau, margin=margin)  # 用户 vs 其余
    L_sph = sphere_reg(encoded)                              # 单位球

    Loss = w_var * L_var + w_con * L_con + w_sph * L_sph
    # 为了兼容你原先的绘图接口，返回三个数
    return Loss, L_var, L_con


# M2 loss
def M2_loss(encoded, y, Lambda):
    L1 = distance_L1(encoded, y)
    L2 = mindistance_L2(encoded, y)
    Loss = L1 + Lambda *L2 #超参数调整

    return Loss, L1, L2