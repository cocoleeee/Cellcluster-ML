import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

def plot_similar_n_compare(
    adata_before,
    adata_after,
    cluster_col_before='orig_annot',
    cluster_col_after='cluster',
    max_n=20,
    metric='cosine',
    figsize=(7,5),
    title='Similar-N: before vs after'
):
    """
    同一数据的表达空间相似度邻居上，分别计算 before/after 两套聚类标签的 Similar-N，
    并在同一张图里绘制对比曲线。
    """
    # --- 邻居在表达空间上统一计算（保证可比性） ---
    X = _to_dense(adata_before.X)
    assert X.shape[0] == adata_after.n_obs and np.all(adata_before.obs_names == adata_after.obs_names), \
        "before/after 的 obs 顺序必须一致"

    nbrs = NearestNeighbors(n_neighbors=max_n+1, metric=metric).fit(X)
    _, idx = nbrs.kneighbors(X)
    neighbor_idx = idx[:, 1:]  # 去掉自己

    # --- 标签取出 ---
    lab_b = adata_before.obs[cluster_col_before].astype(str).values
    lab_a = adata_after.obs [cluster_col_after ].astype(str).values

    # --- 曲线 ---
    xs = np.arange(1, max_n+1)
    y_b, y_a = [], []
    for n in xs:
        match_b = (lab_b[neighbor_idx[:, :n]] == lab_b[:, None]).mean(axis=1).mean()
        match_a = (lab_a[neighbor_idx[:, :n]] == lab_a[:, None]).mean(axis=1).mean()
        y_b.append(float(match_b)); y_a.append(float(match_a))

    # --- 画图 ---
    plt.figure(figsize=figsize)
    plt.plot(xs, y_b, marker='o', label=f'before: {cluster_col_before}')
    plt.plot(xs, y_a, marker='s', label=f'after : {cluster_col_after}')
    plt.xlabel('N (neighbors in expression space)')
    plt.ylabel('平均同簇比例')
    plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return {'x': xs.tolist(), 'before': y_b, 'after': y_a}



import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import pandas as pd

def _ensure_bool(series):
    if series.dtype == bool:
        return series.values
    return series.astype(str).str.lower().isin({'true','1','yes','y','selected'}).values

def _user_related_clusters_topfrac(adata, cluster_col, user_col, top_frac=0.30):
    """按照‘用户细胞计数’排序取前 top_frac 的簇集合（至少 1 个）"""
    df = adata.obs[[cluster_col, user_col]].copy()
    df[cluster_col] = df[cluster_col].astype(str)
    df[user_col] = _ensure_bool(df[user_col])
    counts = (df[df[user_col]].groupby(cluster_col).size().sort_values(ascending=False))
    all_clusters = df[cluster_col].astype(str).unique()
    if counts.empty:
        return set(all_clusters[:1].tolist())
    k = max(1, int(np.ceil(len(all_clusters) * float(top_frac))))
    return set(counts.index[:k].astype(str).tolist())

def compute_dist_metrics_compare(
    adata_before,
    adata_after,
    cluster_col_before='orig_annot',
    cluster_col_after ='cluster',
    user_col='user_selected',     # 可为 bool/categorical/str
    top_frac=0.30,
    figsize=(7,5),
    title='Dist metrics: before vs after',
    plot=True
):
    """
    在各自的 UMAP 上，比较 before/after 的“用户相关簇 vs 非相关簇”的四个距离指标，
    并用并排条形图展示。
    """
    assert 'X_umap' in adata_before.obsm and 'X_umap' in adata_after.obsm, "before/after 均需先有 UMAP 坐标"

    def _metrics(adata, cluster_col):
        labels = adata.obs[cluster_col].astype(str)
        user_clusters = _user_related_clusters_topfrac(adata, cluster_col, user_col, top_frac=top_frac)
        idx1 = np.flatnonzero(labels.isin(list(user_clusters)))
        idx2 = np.flatnonzero(~labels.isin(list(user_clusters)))
        if idx1.size == 0 or idx2.size == 0:
            return {'centroid_dist': np.nan, 'rmse_cross': np.nan,
                    'median_euc_cross': np.nan, 'mean_cosine_cross': np.nan}
        U = adata.obsm['X_umap']
        emb1, emb2 = U[idx1], U[idx2]
        c1, c2 = emb1.mean(axis=0), emb2.mean(axis=0)
        centroid_dist = np.linalg.norm(c1 - c2)
        D_euc = pairwise_distances(emb1, emb2, metric='euclidean')
        rmse_cross = np.sqrt((D_euc**2).mean())
        median_euc_cross = np.median(D_euc)
        D_cos = pairwise_distances(emb1, emb2, metric='cosine')
        mean_cosine_cross = D_cos.mean()
        return {'centroid_dist': centroid_dist, 'rmse_cross': rmse_cross,
                'median_euc_cross': median_euc_cross, 'mean_cosine_cross': mean_cosine_cross}

    m_b = _metrics(adata_before, cluster_col_before)
    m_a = _metrics(adata_after , cluster_col_after )

    if plot:
        names = ['centroid_dist','rmse_cross','median_euc_cross','mean_cosine_cross']
        xb = np.arange(len(names))
        w = 0.38
        plt.figure(figsize=figsize)
        plt.bar(xb - w/2, [m_b[k] for k in names], width=w, label=f'before: {cluster_col_before}')
        plt.bar(xb + w/2, [m_a[k] for k in names], width=w, label=f'after : {cluster_col_after}')
        plt.xticks(xb, names, rotation=25, ha='right')
        plt.ylabel('Value'); plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

    return {'before': m_b, 'after': m_a}


# def plot_nearest_n(adata, n_list=None, cluster_col='cluster', user_col='user_selected',
#                    tau=0.2, min_count=1, figsize=(6,4), user_clusters=None):
#     if n_list is None:
#         n_list = list(range(50, 1001, 50))

#     X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
#     labels = adata.obs[cluster_col].astype(str).values
#     sel_idx = np.flatnonzero(adata.obs[user_col].astype(bool))
#     assert sel_idx.size > 0, "没有用户选择细胞(user_selected=True)。"
#     mean_vec = X[sel_idx].mean(axis=0).reshape(1, -1)

#     # 如果未显式传入上一指标的结果，这里按占比阈值回退一次
#     if user_clusters is None:
#         user_clusters = _user_related_clusters_by_frac(
#             adata, cluster_col=cluster_col, user_col=user_col, tau=tau, min_count=min_count
#         )
#     user_clusters = np.array(list(user_clusters), dtype=str)

#     # 为了 mask 能拿到足够非用户近邻，多取一些邻居
#     extra = len(sel_idx) + 50  # 余量
#     K = min(X.shape[0], max(n_list) + extra)
#     nbrs = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(X)
#     _, idx = nbrs.kneighbors(mean_vec)
#     idx = idx[0]

#     res_nomask, res_mask = [], []
#     sel_set = set(sel_idx)

#     for n in n_list:
#         # womask：直接用前 n 个
#         topn = idx[:n]
#         prop_nomask = np.isin(labels[topn], user_clusters).mean()
#         res_nomask.append(prop_nomask)

#         # mask：按顺序取“前 n 个非用户”
#         mask_list = []
#         for j in idx:
#             if j not in sel_set:
#                 mask_list.append(j)
#                 if len(mask_list) == n:
#                     break
#         if len(mask_list) < n:
#             # 极端情况下不够 n 个非用户，就按实际数量算，也可选择返回 np.nan
#             prop_mask = np.isin(labels[mask_list], user_clusters).mean() if mask_list else np.nan
#         else:
#             prop_mask = np.isin(labels[mask_list], user_clusters).mean()
#         res_mask.append(prop_mask)

#     # 可选调试输出
#     print("user_clusters:", user_clusters, " (count:", len(user_clusters), ")")
#     print("top-20 cluster labels:", labels[idx[:20]])

#     # 绘图
#     plt.figure(figsize=figsize)
#     plt.plot(n_list, res_nomask, marker='o', label='womask（包含用户细胞）')
#     plt.plot(n_list, res_mask, marker='s', label='mask（排除用户细胞）')
#     plt.xlabel('N (neighbors)')
#     plt.ylabel('落入“用户相关簇”的比例')
#     plt.title(f'Nearest-N ({cluster_col}) [tau={tau}, min_count={min_count}]')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     return n_list, {'womask': res_nomask, 'mask': res_mask}

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd

def nearestN_utilization_expression_compare(
    adata_before,
    adata_after,
    cluster_col_before='orig_annot',
    cluster_col_after ='cluster',
    user_col='user_selected',
    n_list=None,                    # 默认 50..1000 步长 50
    metric='cosine',
    extra_buffer=50,
    top_frac=0.30,
    figsize=(7,5),
    title='Nearest-N (expr): before vs after'
):
    """
    在表达空间里，以“用户表达均值”为查询，分别用 before/after 的聚类定义‘用户相关簇’，并
    绘制两套曲线（各自含 womask/mask）。
    """
    if n_list is None:
        n_list = list(range(50, 1001, 50))

    # 统一表达空间（用 adata_before.X；两者应同样本顺序）
    X = _to_dense(adata_before.X)
    assert X.shape[0] == adata_after.n_obs and np.all(adata_before.obs_names == adata_after.obs_names), \
        "before/after 的 obs 顺序必须一致"

    labels_b = adata_before.obs[cluster_col_before].astype(str).values
    labels_a = adata_after.obs [cluster_col_after ].astype(str).values
    sel_bool = _ensure_bool(adata_before.obs[user_col])
    sel_idx = np.flatnonzero(sel_bool)
    assert sel_idx.size > 0, "没有用户选择细胞（user_selected=True）。"

    # —— 定义“用户相关簇”：前 top_frac 的高计数簇（各自的标签空间内定义）——
    def _user_clusters(labels):
        s = pd.Series(labels[sel_idx])
        counts = s.value_counts()
        # 所有簇集合（防止没出现的簇）
        all_clusters = pd.Index(pd.Series(labels).unique().astype(str))
        if counts.empty:
            return set(all_clusters[:1].tolist())
        k = max(1, int(np.ceil(len(all_clusters) * float(top_frac))))
        return set(counts.index[:k].astype(str).tolist())

    uc_b = _user_clusters(labels_b)
    uc_a = _user_clusters(labels_a)
    print(f"[Nearest-N] before user-clusters(top {int(top_frac*100)}%): {list(uc_b)}")
    print(f"[Nearest-N] after  user-clusters(top {int(top_frac*100)}%): {list(uc_a)}")

    # —— 最近邻查询（表达空间）——
    K = min(X.shape[0], max(n_list) + sel_idx.size + extra_buffer)
    nn = NearestNeighbors(n_neighbors=K, metric=metric).fit(X)
    _, idx = nn.kneighbors(X[sel_idx].mean(axis=0, keepdims=True))
    idx = idx[0]

    sel_set = set(sel_idx)
    def _curves(user_clusters):
        womask, mask = [], []
        for n in n_list:
            # womask：直接前 n
            topn = idx[:n]
            womask.append(float(np.isin(labels_b[topn], list(user_clusters)).mean()))  # 用任意一套 labels 判断簇归属
            # mask：按顺序取 n 个非用户
            mask_list = []
            for j in idx:
                if j not in sel_set:
                    mask_list.append(j)
                    if len(mask_list) == n:
                        break
            mask.append(float(np.isin(labels_b[mask_list], list(user_clusters)).mean()) if len(mask_list)==n else np.nan)
        return womask, mask

    wom_b, msk_b = _curves(uc_b)
    wom_a, msk_a = _curves(uc_a)

    # —— 画图（同图对比）——
    plt.figure(figsize=figsize)
    plt.plot(n_list, wom_b, marker='o',  label=f'before-womask ({cluster_col_before})')
    plt.plot(n_list, msk_b, marker='s',  label=f'before-mask   ({cluster_col_before})')
    plt.plot(n_list, wom_a, marker='^',  label=f'after-womask  ({cluster_col_after})')
    plt.plot(n_list, msk_a, marker='x',  label=f'after-mask    ({cluster_col_after})')
    plt.xlabel('N (neighbors in expression space)')
    plt.ylabel('落入“用户相关簇”的比例')
    plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return {
        'n_list': n_list,
        'before': {'womask': wom_b, 'mask': msk_b, 'user_clusters': list(uc_b)},
        'after' : {'womask': wom_a, 'mask': msk_a, 'user_clusters': list(uc_a)}
    }
