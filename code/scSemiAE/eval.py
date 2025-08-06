import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# 评估指标模块: Similar-N, Dist, Nearest-N

def plot_similar_n(adata, max_n=20, figsize=(6,4)):
    """
    计算并绘制 Similar-N 曲线：对每个细胞，取余弦相似度最高的前 n 个细胞聚类一致比例，n=1..max_n。
    返回 proportions 列表。
    """
    # 1 - cosine 作为距离
    nbrs = NearestNeighbors(n_neighbors=max_n+1, metric='cosine').fit(adata.X)
    dist, idx = nbrs.kneighbors(adata.X)
    neighbor_idx = idx[:, 1:]
    labels = adata.obs['cluster'].values

    proportions = []
    for n in range(1, max_n+1):
        match = (labels[neighbor_idx[:, :n]] == labels[:, None])
        prop = match.mean(axis=1).mean()
        proportions.append(prop)

    # 绘图
    plt.figure(figsize=figsize)
    plt.plot(range(1, max_n+1), proportions, marker='o')
    plt.xlabel('N (neighbors)')
    plt.ylabel('平均同簇比例')
    plt.title('Similar-N 曲线')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("similar_n.png")
    plt.show()

    return proportions


def compute_dist_metrics(adata, plot=True, figsize=(6,4)):
    """
    计算 Dist 指标并可选绘制：
      - 重心距离 (centroid_dist)
      - 平均 RMSE (rmse_cross)
      - 中位数欧氏距离 (median_euc_cross)
      - 平均余弦距离 (mean_cosine_cross)
    返回字典 metrics。
    """
    umap = adata.obsm['X_umap']
    labels = adata.obs['cluster']
    sel = adata.obs['user_selected']

    # 用户相关簇
    user_clusters = labels[sel].unique()
    idx1 = np.flatnonzero(labels.isin(user_clusters))
    idx2 = np.flatnonzero(~labels.isin(user_clusters))
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
        plt.title('Dist 指标')
        plt.tight_layout()
        plt.savefig("dist_metrics.png")
        plt.show()

    return metrics


def plot_nearest_n(adata, n_list=None, figsize=(6,4)):
    """
    计算并绘制 Nearest-N 曲线：基于用户选择细胞平均表达，寻找最相近前 n 个细胞，统计落入用户相关聚类的比例。
    返回 n_list, results 字典。
    """
    if n_list is None:
        n_list = list(range(50, 1001, 50))

    # 准备数据
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    sel_idx = np.flatnonzero(adata.obs['user_selected'])
    mean_vec = X[sel_idx].mean(axis=0).reshape(1, -1)
    nbrs = NearestNeighbors(n_neighbors=max(n_list), metric='euclidean').fit(X)
    _, idx = nbrs.kneighbors(mean_vec)
    idx = idx[0]

    labels = adata.obs['cluster'].values
    user_clusters = labels[sel_idx]
    user_clusters = np.unique(user_clusters)

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
    plt.title('Nearest-N 曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("nearest_n.png")
    plt.show()

    results = {'nomask': res_nomask, 'mask': res_mask}
    return n_list, results
