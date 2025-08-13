import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def plot_similar_n(adata, max_n=20, cluster_col='cluster', figsize=(6,4)):
    """
    计算并绘制 Similar-N 曲线：对每个细胞，取余弦相似度最高的前 n 个细胞聚类一致比例，n=1..max_n。

    参数:
        adata: AnnData 对象，需包含表达矩阵 adata.X 和聚类标签列 adata.obs[cluster_col]
        max_n: 最多邻居数 N
        cluster_col: 用于评估的聚类标签所在列名
        figsize: 图形大小
    返回:
        proportions: 长度为 max_n 的列表，每个元素是对应 N 的平均同簇比例
    """
    labels = adata.obs[cluster_col].astype(str).values
    # 邻居计算，使用 1-cosine 作为距离
    nbrs = NearestNeighbors(n_neighbors=max_n+1, metric='cosine').fit(adata.X)
    dist, idx = nbrs.kneighbors(adata.X)
    neighbor_idx = idx[:, 1:]

    proportions = []
    for n in range(1, max_n+1):
        match = (labels[neighbor_idx[:, :n]] == labels[:, None])
        proportions.append(match.mean(axis=1).mean())

    # 绘图
    plt.figure(figsize=figsize)
    plt.plot(range(1, max_n+1), proportions, marker='o')
    plt.xlabel('N (neighbors)')
    plt.ylabel('平均同簇比例')
    plt.title(f'Similar-N ({cluster_col})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return proportions


def _user_related_clusters_by_frac(adata, cluster_col='cluster', user_col='user_selected',
                                   tau=0.2, min_count=1):
    """
    返回满足条件的簇集合：
    条件 = 该簇内 user_selected==True 的比例 >= tau 且 该簇内 True 个数 >= min_count
    若没有任何簇满足且样本中存在 True，则回退为“占比最高的 Top-1”。
    """
    grp = adata.obs.groupby(cluster_col)[user_col]
    frac = grp.mean()                          # 每簇 True 的比例
    cnt  = grp.sum()                           # 每簇 True 的个数
    mask = (frac >= tau) & (cnt >= min_count)
    user_clusters = frac.index[mask]

    # 回退：若有人被选中但没有簇达标，则取占比最高的 Top-1 作为相关簇
    if len(user_clusters) == 0 and (adata.obs[user_col].astype(bool).sum() > 0):
        top1 = frac.sort_values(ascending=False).index[:3]
        user_clusters = top1
    return set(map(str, user_clusters))  # 统一成 str，和下游 labels 比较一致



def compute_dist_metrics(adata, cluster_col='cluster', user_col='user_selected',
                         tau=0.01, min_count=1, plot=True, figsize=(6,4)):
    """
    Dist 指标：基于 UMAP，比较“用户相关簇(>=tau)” vs “非相关簇”。
    新增参数:
        tau: 把簇判为用户相关的占比阈值 (默认 0.2)
        min_count: 该簇内用户细胞的最小计数（默认 1）
    """
    labels = adata.obs[cluster_col].astype(str)
    umap = adata.obsm['X_umap']

    user_clusters = _user_related_clusters_by_frac(
        adata, cluster_col=cluster_col, user_col=user_col, tau=tau, min_count=min_count
    )
    idx1 = np.flatnonzero(labels.isin(list(user_clusters)))
    idx2 = np.flatnonzero(~labels.isin(list(user_clusters)))

    # 无法分组时返回 NaN
    if idx1.size == 0 or idx2.size == 0:
        print(f"[Dist] Warning: 无法区分用户相关与非相关簇（tau={tau}, min_count={min_count}）。返回 NaN。")
        nan_metrics = {k: np.nan for k in ['centroid_dist','rmse_cross','median_euc_cross','mean_cosine_cross']}
        if plot:
            print("跳过绘图。")
        return nan_metrics

    emb1, emb2 = umap[idx1], umap[idx2]
    c1, c2 = emb1.mean(axis=0), emb2.mean(axis=0)
    centroid_dist = np.linalg.norm(c1 - c2)

    D_euc = pairwise_distances(emb1, emb2, metric='euclidean')
    rmse_cross = np.sqrt((D_euc**2).mean())
    median_euc_cross = np.median(D_euc)

    D_cos = pairwise_distances(emb1, emb2, metric='cosine')
    mean_cosine_cross = D_cos.mean()

    metrics = {
        'centroid_dist': centroid_dist,
        'rmse_cross': rmse_cross,
        'median_euc_cross': median_euc_cross,
        'mean_cosine_cross': mean_cosine_cross
    }

    if plot:
        names = list(metrics.keys())
        values = [metrics[k] for k in names]
        plt.figure(figsize=figsize)
        plt.bar(names, values)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Value')
        plt.title(f'Dist ({cluster_col}) [tau={tau}, min_count={min_count}]')
        plt.tight_layout()
        plt.show()

    return metrics

def plot_nearest_n(adata, n_list=None, cluster_col='cluster', user_col='user_selected',
                   tau=0.2, min_count=1, figsize=(6,4), user_clusters=None):
    if n_list is None:
        n_list = list(range(50, 1001, 50))

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    labels = adata.obs[cluster_col].astype(str).values
    sel_idx = np.flatnonzero(adata.obs[user_col].astype(bool))
    assert sel_idx.size > 0, "没有用户选择细胞(user_selected=True)。"
    mean_vec = X[sel_idx].mean(axis=0).reshape(1, -1)

    # 如果未显式传入上一指标的结果，这里按占比阈值回退一次
    if user_clusters is None:
        user_clusters = _user_related_clusters_by_frac(
            adata, cluster_col=cluster_col, user_col=user_col, tau=tau, min_count=min_count
        )
    user_clusters = np.array(list(user_clusters), dtype=str)

    # 为了 mask 能拿到足够非用户近邻，多取一些邻居
    extra = len(sel_idx) + 50  # 余量
    K = min(X.shape[0], max(n_list) + extra)
    nbrs = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(X)
    _, idx = nbrs.kneighbors(mean_vec)
    idx = idx[0]

    res_nomask, res_mask = [], []
    sel_set = set(sel_idx)

    for n in n_list:
        # womask：直接用前 n 个
        topn = idx[:n]
        prop_nomask = np.isin(labels[topn], user_clusters).mean()
        res_nomask.append(prop_nomask)

        # mask：按顺序取“前 n 个非用户”
        mask_list = []
        for j in idx:
            if j not in sel_set:
                mask_list.append(j)
                if len(mask_list) == n:
                    break
        if len(mask_list) < n:
            # 极端情况下不够 n 个非用户，就按实际数量算，也可选择返回 np.nan
            prop_mask = np.isin(labels[mask_list], user_clusters).mean() if mask_list else np.nan
        else:
            prop_mask = np.isin(labels[mask_list], user_clusters).mean()
        res_mask.append(prop_mask)

    # 可选调试输出
    print("user_clusters:", user_clusters, " (count:", len(user_clusters), ")")
    print("top-20 cluster labels:", labels[idx[:20]])

    # 绘图
    plt.figure(figsize=figsize)
    plt.plot(n_list, res_nomask, marker='o', label='womask（包含用户细胞）')
    plt.plot(n_list, res_mask, marker='s', label='mask（排除用户细胞）')
    plt.xlabel('N (neighbors)')
    plt.ylabel('落入“用户相关簇”的比例')
    plt.title(f'Nearest-N ({cluster_col}) [tau={tau}, min_count={min_count}]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return n_list, {'womask': res_nomask, 'mask': res_mask}
