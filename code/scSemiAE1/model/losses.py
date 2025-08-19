# losses.py
import torch
import torch.nn.functional as F

# ------------------------ Utils ------------------------

@torch.no_grad()
def _make_user_mask_from_cells(cells, user_set, device):
    """cells: List[str], user_set: Set[str] -> BoolTensor[N]"""
    if cells is None or user_set is None:
        return None
    m = [(c in user_set) for c in cells]
    if not any(m):
        return None
    return torch.tensor(m, dtype=torch.bool, device=device)

def _remap_labels(y: torch.Tensor) -> torch.Tensor:
    """
    返回一个不修改输入 y 的 0..K-1 重映射副本（避免 in-place 破坏计算图/梯度期望）。
    """
    y = y.detach().clone()
    uniq = torch.unique(y, sorted=True)
    remap = torch.empty_like(y)
    for new_id, v in enumerate(uniq):
        remap[y == v] = new_id
    return remap.long()

def _class_centers(z: torch.Tensor, y01: torch.Tensor):
    """
    z: (N,D), y01: (N,) 已重映射到 0..K-1
    return: centers (K,D), counts (K,)
    """
    K = int(y01.max().item()) + 1
    D = z.size(1)
    centers = z.new_zeros(K, D)
    counts  = z.new_zeros(K, dtype=torch.long)
    for c in range(K):
        idx = (y01 == c).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            centers[c] = z.index_select(0, idx).mean(0)
            counts[c] = idx.numel()
    return centers, counts

def distance_L1(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    类内紧致度（越小越好）: 每个样本到其类别中心的平方距离的平均。
    """
    y01 = _remap_labels(y)
    centers, _ = _class_centers(z, y01)  # (K,D)
    dif = z - centers.index_select(0, y01)
    return (dif * dif).sum(dim=1).mean()

def mindistance_L2_pos(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    类间最小中心距（正数，越大越好）：
    计算类别中心两两欧氏距离的最小值；如果 K<=1，返回 0。
    """
    y01 = _remap_labels(y)
    centers, _ = _class_centers(z, y01)  # (K,D)
    K = centers.size(0)
    if K <= 1:
        return centers.new_tensor(0.0)
    dmat = torch.cdist(centers, centers, p=2)  # (K,K)
    mask = ~torch.eye(K, dtype=torch.bool, device=z.device)
    # 所有非对角距离的最小值
    min_d = dmat[mask].min()
    return min_d




# ------------------------ Main Loss ------------------------

def M2_loss(
    z, y, Lambda=1,
    cells=None, user_set=None,
    # 个性化项 + 正则
    w_pull=0.3, w_push=0.3, margin=5, w_reg=1e-2,
):
    """
    总体目标 = 全局判别(原 L1/L2) + 个性化(user pull/push) + 轻正则

    - 全局 L1（类内）：  distance_L1(z, y)             —— 越小越好
    - 全局 L2（类间）：  mindistance_L2_pos(z, y)       —— 越大越好（我们在主目标中做 -Lambda*L2）
    - 用户 L_pull：      用户样本收紧到用户质心
    - 用户 L_push：      非用户远离用户质心（至少 margin）
    - L_reg：            防坍缩/数值稳定的轻正则
    """
    device = z.device

    # --- 建议：全局判别用“原始 z”，个性化用“单位化 z_norm”（更像余弦距离）
    z_raw  = z
    z_norm = F.normalize(z, dim=1)

    # ========= 1) 全局判别（原 L1 / L2） =========
    L1 = distance_L1(z_raw, y)             # 小 -> 好
    L2 = mindistance_L2_pos(z_raw, y)      # 大 -> 好
    L_main = L1 - Lambda * L2             

    # ========= 2) 个性化（用户拉近 / 推远） =========
    user_mask = _make_user_mask_from_cells(cells, user_set, device) if cells is not None else None

    if (user_mask is not None) and (user_mask.sum().item() >= 2):
        z_u = z_norm[user_mask]
        z_b = z_norm[~user_mask] if (~user_mask).any() else None

        c_u = z_u.mean(dim=0, keepdim=True)   # (1,D)

        # 拉近：用户样本 -> 用户质心
        L_pull = ((z_u - c_u).pow(2).sum(dim=1)).mean()

        # 推远：非用户 至少 margin
        if z_b is not None and z_b.size(0) > 0:
            d = torch.cdist(z_b, c_u, p=2).squeeze(1)  # (Nb,)
            L_push = F.relu(margin - d).mean()
        else:
            L_push = z.new_tensor(0.0)
    else:
        # 没有用户样本（或不足 2 个），个性化项置零
        L_pull = z.new_tensor(0.0)
        L_push = z.new_tensor(0.0)

    # ========= 3) 轻正则 =========
    L_reg = (z_norm.pow(2).sum(dim=1) - 1.0).abs().mean()  # 理论上 ~0

    # ========= 4) 汇总 =========
    total = L_main +(w_pull * L_pull + w_push * L_push + w_reg * L_reg)

    comps = {
        # 原始三件套
        'L1':        float(L1.detach().cpu()),
        'L2':        float(L2.detach().cpu()),
        # 用户两项
        'L_pull':   float(L_pull.detach().cpu()),
        'L_push':   float(L_push.detach().cpu()),

        'L_total':   float(total.detach().cpu()),
    }
    return total, comps
