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


def compute_dist_metrics(adata, cluster_col='cluster', user_col='user_selected', plot=True, figsize=(6,4)):
    """
    计算 Dist 指标：基于 UMAP 坐标，评估用户相关聚类与其他聚类的分离度。

    参数:
        adata: 包含 adata.obsm['X_umap'] 和标签列
        cluster_col: 聚类标签列名
        user_col: 用户选择细胞布尔列名
        plot: 是否绘图
        figsize: 图形大小
    返回:
        metrics: 包含四项指标的字典。如果只有一类或没有“非相关”细胞，则所有指标返回 NaN 并打印警告。
    """
    labels = adata.obs[cluster_col].astype(str)
    sel = adata.obs[user_col].astype(bool)
    umap = adata.obsm['X_umap']

    # 用户相关簇
    user_clusters = labels[sel].unique()
    idx1 = np.flatnonzero(labels.isin(user_clusters))
    idx2 = np.flatnonzero(~labels.isin(user_clusters))

    # 如果没有分组或某组为空，无法计算
    if idx1.size == 0 or idx2.size == 0:
        print(f"Warning: 无法区分用户相关与非相关细胞，可能聚类('{cluster_col}')只有一类或所有细胞都在同一簇。返回 NaN。")
        nan_metrics = {k: np.nan for k in ['centroid_dist','rmse_cross','median_euc_cross','mean_cosine_cross']}
        if plot:
            print("跳过绘图。")
        return nan_metrics

    emb1, emb2 = umap[idx1], umap[idx2]

    # 重心距离
    c1, c2 = emb1.mean(axis=0), emb2.mean(axis=0)
    centroid_dist = np.linalg.norm(c1 - c2)
    # 跨组欧氏距离矩阵
    D_euc = pairwise_distances(emb1, emb2, metric='euclidean')
    rmse_cross = np.sqrt((D_euc**2).mean())
    median_euc_cross = np.median(D_euc)
    # 跨组余弦距离
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
        plt.title(f'Dist ({cluster_col})')
        plt.tight_layout()
        plt.show()

    return metrics

def plot_nearest_n(adata, n_list=None, cluster_col='cluster', user_col='user_selected', figsize=(6,4)):
    """
    计算并绘制 Nearest-N 曲线：基于用户选择细胞平均表达，寻找最相近前 n 个细胞，统计落入用户相关聚类的比例。

    参数:
        adata: 包含 adata.X, adata.obs[cluster_col], adata.obs[user_col]
        n_list: 可选的 N 列表，默认 50..1000 步长50
        cluster_col: 聚类标签列名
        user_col: 用户选择列名
        figsize: 图形大小
    返回:
        n_list, results: 结果字典包含 'nomask' 和 'mask'
    """
    if n_list is None:
        n_list = list(range(50, 1001, 50))

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    labels = adata.obs[cluster_col].astype(str).values
    sel_idx = np.flatnonzero(adata.obs[user_col].astype(bool))
    mean_vec = X[sel_idx].mean(axis=0).reshape(1, -1)

    nbrs = NearestNeighbors(n_neighbors=max(n_list), metric='euclidean').fit(X)
    _, idx = nbrs.kneighbors(mean_vec)
    idx = idx[0]

    user_clusters = np.unique(labels[sel_idx])
    res_nomask, res_mask = [], []
    sel_set = set(sel_idx)
    for n in n_list:
        topn = idx[:n]
        prop_nomask = np.isin(labels[topn], user_clusters).mean()
        filtered = [i for i in topn if i not in sel_set]
        prop_mask = np.isin(labels[filtered], user_clusters).mean() if filtered else np.nan
        res_nomask.append(prop_nomask)
        res_mask.append(prop_mask)

    # 绘图
    plt.figure(figsize=figsize)
    plt.plot(n_list, res_nomask, marker='o', label='包含用户细胞')
    plt.plot(n_list, res_mask, marker='s', label='排除用户细胞')
    plt.xlabel('N (neighbors)')
    plt.ylabel('落入用户相关聚类的比例')
    plt.title(f'Nearest-N ({cluster_col})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    results = {'nomask': res_nomask, 'mask': res_mask}
    return n_list, results
