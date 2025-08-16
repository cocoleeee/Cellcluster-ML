# losses.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def _make_user_mask_from_cells(cells, user_set, device):
    """
    cells: list-like of cell ids (str) for current batch
    user_set: set of user-selected cell ids (str)
    return: BoolTensor mask on device, shape (N,)
    """
    if cells is None or user_set is None:
        return None
    mask = [ (c in user_set) for c in cells ]
    if not any(mask):
        return None
    return torch.tensor(mask, dtype=torch.bool, device=device)

def _compute_centers(z, y, n_classes):
    """
    z: (N,D), y: (N,)
    return centers: (K,D)
    """
    D = z.size(1)
    centers = z.new_zeros((n_classes, D))
    for c in range(n_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            centers[c] = z.index_select(0, idx).mean(0)
    return centers

def _intra_class_loss(z, y, centers):
    """
    L1: mean squared L2-to-center
    """
    dif = z - centers.index_select(0, y)
    return (dif * dif).sum(dim=1).mean()

def _inter_class_separation(centers, softmin_tau=0.5, eps=1e-8):
    """
    L2: soft-min of pairwise center distances (squared)
    Larger is better -> will be subtracted in objective.
    """
    K = centers.size(0)
    if K <= 1:
        return centers.new_tensor(0.0)
    dmat = torch.cdist(centers, centers, p=2)  # (K,K)
    mask = torch.eye(K, device=centers.device, dtype=torch.bool)
    dvec = dmat[~mask]  # K*(K-1)
    if dvec.numel() == 0:
        return centers.new_tensor(0.0)
    w = torch.softmax(-dvec / max(softmin_tau, eps), dim=0)
    softmin = (w * dvec).sum()
    return softmin * softmin

def _user_inter_separation(centers, user_classes, softmin_tau=0.5, eps=1e-8):
    """
    L2_user: 对每个与用户相关的类别中心，计算它到“其他中心”的 soft-min 距离（平方），再求平均。
    关键：不用任何 in-place 改写，完全用掩码切片。
    """
    if user_classes is None or (isinstance(user_classes, torch.Tensor) and user_classes.numel() == 0):
        return centers.new_tensor(0.0)

    # pairwise distances
    dmat = torch.cdist(centers, centers, p=2)  # (K, K)
    K = dmat.size(0)
    device = dmat.device

    out = []
    for c in (user_classes.tolist() if isinstance(user_classes, torch.Tensor) else list(user_classes)):
        # 选取“除自身外”的其它中心
        mask = torch.ones(K, dtype=torch.bool, device=device)
        mask[c] = False
        others = dmat[c][mask]  # (K-1,)
        if others.numel() == 0:
            continue
        w = torch.softmax(-others / max(softmin_tau, eps), dim=0)
        sm = (w * others).sum()
        out.append(sm * sm)
    if not out:
        return centers.new_tensor(0.0)
    return torch.stack(out).mean()




def _triplet_loss_user(z, y, user_mask, margin=0.5):
    """
    Batch-level triplet on user subset:
    Anchor-positive same label, negative = closest negative.
    """
    idx = user_mask.nonzero(as_tuple=True)[0]
    if idx.numel() < 3:
        return z.new_tensor(0.0)
    z_u, y_u = z.index_select(0, idx), y.index_select(0, idx)
    dmat = torch.cdist(z_u, z_u, p=2)
    M = dmat.size(0)
    losses = []
    for i in range(M):
        pos_pool = (y_u == y_u[i]).nonzero(as_tuple=True)[0]
        neg_pool = (y_u != y_u[i]).nonzero(as_tuple=True)[0]
        if pos_pool.numel() <= 1 or neg_pool.numel() == 0:
            continue
        # pick a positive not self
        p = pos_pool[torch.randint(0, pos_pool.numel(), (1,))]
        if p.item() == i and pos_pool.numel() > 1:
            p = pos_pool[(1 + (pos_pool == p).nonzero(as_tuple=True)[0]) % pos_pool.numel()]
        # hardest negative (closest)
        dn_row = dmat[i].index_select(0, neg_pool)
        n = neg_pool[dn_row.argmin()]
        da = dmat[i, p]
        dn = dmat[i, n]
        losses.append(F.relu(da - dn + margin))
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def _supcon_user(z, y, user_mask, tau=0.1, eps=1e-8):
    """
    Supervised contrastive（欧氏距离→相似度）用户子集版本，无 in-place。
    """
    idx = user_mask.nonzero(as_tuple=True)[0]
    if idx.numel() <= 2:
        return z.new_tensor(0.0)
    zu, yu = z.index_select(0, idx), y.index_select(0, idx)
    dmat = torch.cdist(zu, zu, p=2)                      # (M, M)
    sim = torch.exp(-dmat / max(tau, eps))               # (M, M)
    M = sim.size(0)
    eye = torch.eye(M, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, 0.0)                      # 非原地写入

    losses = []
    for i in range(M):
        pos_mask = (yu == yu[i]) & (~eye[i])             # 不再做 pos[i]=False
        denom = sim[i].sum() + eps
        num   = (sim[i][pos_mask]).sum() + eps
        if pos_mask.any():
            losses.append(-(num / denom).log())
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()



def M2_loss(
    z, y, Lambda,
    cells=None,                 # list of cell ids for current batch (可为 None)
    user_set=None,              # set(str) of user-selected cell ids (可为 None)
    alpha_user=3.0,             # 用户类内收紧权重
    beta_user=3.0,              # 用户类间分离权重
    use_triplet=True,  w_triplet=0.1, triplet_margin=0.5,
    use_supcon=True,   w_supcon=0.1,  supcon_tau=0.1,
    softmin_tau=0.5,   eps=1e-8,
):
    """
    论文主目标 + 个性化扩展的总损失。
    返回:
        total_loss, L1_plot, L2_plot
    其中 L1_plot = L1 + alpha_user*L1_user,  L2_plot = L2 + beta_user*L2_user （便于你现有的曲线绘制）
    """
    device = z.device
    # 类别数可用 batch 内最大标签 + 1 近似；若你维护全局 K，可传入而非推断
    n_classes = int(y.max().item()) + 1

    centers = _compute_centers(z, y, n_classes)
    L1 = _intra_class_loss(z, y, centers)
    L2 = _inter_class_separation(centers, softmin_tau=softmin_tau, eps=eps)

    # 用户相关部分
    user_mask = _make_user_mask_from_cells(cells, user_set, device) if (cells is not None) else None
    if user_mask is not None:
        zu = z[user_mask]
        yu = y[user_mask]
        # 类内（对用户样本到其各自类别中心）
        dif_u = zu - centers.index_select(0, yu)
        L1_user = (dif_u * dif_u).sum(dim=1).mean()
        # 类间（用户涉及到的类别中心与其他中心的分离）
        user_classes = torch.unique(yu)
        L2_user = _user_inter_separation(centers, user_classes, softmin_tau=softmin_tau, eps=eps)
    else:
        L1_user = z.new_tensor(0.0)
        L2_user = z.new_tensor(0.0)

    # 主目标
    L1_total = L1 + alpha_user * L1_user
    L2_total = L2 + beta_user * L2_user
    L_main = L1_total - Lambda * L2_total

    # 可选：只在用户子集上做三元组 / 对比
    L_tri = z.new_tensor(0.0)
    L_con = z.new_tensor(0.0)
    if user_mask is not None:
        if use_triplet:
            L_tri = _triplet_loss_user(z, y, user_mask, margin=triplet_margin)
        if use_supcon:
            L_con = _supcon_user(z, y, user_mask, tau=supcon_tau, eps=eps)

    total = L_main + w_triplet * L_tri + w_supcon * L_con
    # 按你当前绘图逻辑返回 L1_total / L2_total
    return total, L1_total, L2_total
