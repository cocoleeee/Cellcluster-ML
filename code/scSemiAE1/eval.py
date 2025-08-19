import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pathlib import Path


def _write_tsv(path, header, rows):
    """把 header + rows 写成 TSV 到 path；自动建目录并打印绝对路径。"""
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        if header:
            f.write("\t".join(map(str, header)) + "\n")
        for r in rows:
            f.write("\t".join(map(lambda x: f"{float(x):.6f}" if isinstance(x, (int, float)) else str(x), r)) + "\n")
    print(f"[Saved plot data] {p.resolve()}")



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

    _write_tsv(
            "similarN.txt",
            header=["N", "before_same_label_ratio", "after_same_label_ratio"],
            rows=zip(xs, y_b, y_a)
        )
    # --- 画图 ---
    plt.figure(figsize=figsize)
    plt.plot(xs, y_b, marker='o', label=f'before: {cluster_col_before}')
    plt.plot(xs, y_a, marker='s', label=f'after : {cluster_col_after}')
    plt.xlabel('N (neighbors in expression space)')
    plt.ylabel('平均同簇比例')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("similarN.png")
    plt.show()

    return {'x': xs.tolist(), 'before': y_b, 'after': y_a}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def _ensure_bool(series):
    if series.dtype == bool:
        return series.values
    return series.astype(str).str.lower().isin({'true','1','yes','y','selected'}).values

def _get_user_clusters_by_proportion(labels, sel_idx, top_frac=0.30, min_clusters=1):
    """
    自动判定“用户相关簇”：按‘用户细胞比例’从高到低取前 top_frac（至少 min_clusters 个），
    过滤掉 user_frac = 0 的簇；并在比例相同的情况下用用户细胞数降序兜底。
    返回：(set(簇名), 统计表)
    """
    lab = pd.Series(labels, name='cluster').astype(str)
    is_user = pd.Series(np.isin(np.arange(len(labels)), sel_idx), name='is_user')
    df = pd.concat([lab, is_user], axis=1)
    grp = df.groupby('cluster')
    cnt_all  = grp['is_user'].size().rename('n_all')
    cnt_user = grp['is_user'].sum().rename('n_user')
    frac = (cnt_user / cnt_all).fillna(0.0).rename('user_frac')

    k = max(min_clusters, int(np.ceil(len(frac) * float(top_frac))))
    order = frac.to_frame().join(cnt_user).sort_values(['user_frac','n_user'], ascending=False)
    keep = order[order['user_frac'] > 0.0].head(k).index.tolist()

    if len(keep) == 0:  # 兜底：若全为 0
        keep = cnt_user.sort_values(ascending=False).head(k).index.tolist()

    print("[User-clusters by proportion] top entries:")
    print(order.head(10).to_string())
    return set(map(str, keep)), order

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def _ensure_bool(series):
    if series.dtype == bool:
        return series.values
    return series.astype(str).str.lower().isin({'true','1','yes','y','selected'}).values

def _get_user_clusters_by_proportion(labels, sel_idx, top_frac=0.30, min_clusters=1):
    lab = pd.Series(labels, name='cluster').astype(str)
    is_user = pd.Series(np.isin(np.arange(len(labels)), sel_idx), name='is_user')
    df = pd.concat([lab, is_user], axis=1)
    grp = df.groupby('cluster')
    cnt_all  = grp['is_user'].size().rename('n_all')
    cnt_user = grp['is_user'].sum().rename('n_user')
    frac = (cnt_user / cnt_all).fillna(0.0).rename('user_frac')

    k = max(min_clusters, int(np.ceil(len(frac) * float(top_frac))))
    order = frac.to_frame().join(cnt_user).sort_values(['user_frac','n_user'], ascending=False)
    keep = order[order['user_frac'] > 0.0].head(k).index.tolist()
    if len(keep) == 0:  # 兜底
        keep = cnt_user.sort_values(ascending=False).head(k).index.tolist()

    print("[User-clusters by proportion] top entries:")
    print(order.head(10).to_string())
    return set(map(str, keep)), order

def _intra_pairwise_stats(emb):
    """组内两两距离的统计：RMSE、median_euc、mean_cos（均在 UMAP 空间）"""
    n = emb.shape[0]
    if n <= 1:
        return np.nan, np.nan, np.nan
    D_euc = pairwise_distances(emb, emb, metric='euclidean')
    iu = np.triu_indices(n, k=1)
    vals = D_euc[iu]
    rmse = float(np.sqrt((vals**2).mean()))
    median_euc = float(np.median(vals))
    D_cos = pairwise_distances(emb, emb, metric='cosine')
    mean_cos = float(D_cos[iu].mean())
    return rmse, median_euc, mean_cos



def compute_dist_metrics_compare(
    adata,
    cluster_col='cluster',
    user_col='user_selected',
    top_frac=0.30,
    title='User-related vs Non-related (UMAP, 4 metrics)',
    figsize=(10,5),
    savepath="dist_user_vs_non_4metrics.png",
    show=True
):
    """
    在同一个 adata 上（需已有 adata.obsm['X_umap']），自动判定“用户相关簇 vs 非相关簇”，
    并分别计算四个指标（组内/相对全局），在一张图里对比：
      1) Centroid-to-Global（各组质心到全体质心的欧氏距离）
      2) Intra RMSE（组内两两欧氏距离 RMSE）
      3) Intra Median-Euclidean（组内两两欧氏距离中位数）
      4) Intra Mean-Cosine（组内两两余弦距离平均）
    """
    assert 'X_umap' in adata.obsm, "需要先在 adata.obsm['X_umap'] 中提供 UMAP 坐标。"
    U = adata.obsm['X_umap']
    labels = adata.obs[cluster_col].astype(str).values
    sel_bool = _ensure_bool(adata.obs[user_col])
    sel_idx  = np.flatnonzero(sel_bool)

    # 自动判定用户相关簇（按占比）
    user_clusters, stat_table = _get_user_clusters_by_proportion(
        labels, sel_idx, top_frac=float(top_frac), min_clusters=1
    )
    print(f"[user-vs-non] user-related clusters (top {int(top_frac*100)}% by proportion): {sorted(list(user_clusters))}")

    # 划分两组索引
    is_user_cluster = np.isin(labels.astype(str), list(user_clusters))
    idx_user = np.flatnonzero(is_user_cluster)
    idx_non  = np.flatnonzero(~is_user_cluster)

    metrics = {}
    if idx_user.size == 0 or idx_non.size == 0:
        print(f"[WARN] one side empty. user={idx_user.size}, non={idx_non.size}")
        metrics.update({
            'centroid_to_global': {'user': np.nan, 'non': np.nan},
            'rmse_intra':         {'user': np.nan, 'non': np.nan},
            'median_euc_intra':   {'user': np.nan, 'non': np.nan},
            'mean_cos_intra':     {'user': np.nan, 'non': np.nan},
        })
    else:
        # 全局质心
        c_all = U.mean(axis=0)

        # 用户组
        emb_user = U[idx_user]
        c_user = emb_user.mean(axis=0)
        d_user_centroid_global = float(np.linalg.norm(c_user - c_all))
        rmse_u, med_u, mcos_u = _intra_pairwise_stats(emb_user)

        # 非用户组
        emb_non = U[idx_non]
        c_non = emb_non.mean(axis=0)
        d_non_centroid_global = float(np.linalg.norm(c_non - c_all))
        rmse_n, med_n, mcos_n = _intra_pairwise_stats(emb_non)

        metrics.update({
            'centroid_to_global': {'user': d_user_centroid_global, 'non': d_non_centroid_global},
            'rmse_intra':         {'user': rmse_u, 'non': rmse_n},
            'median_euc_intra':   {'user': med_u,  'non': med_n},
            'mean_cos_intra':     {'user': mcos_u, 'non': mcos_n},
        })

    # ------------ 画在一张图里（四组，每组两根柱：user vs non）------------
    names = ['Centroid→Global', 'Intra RMSE', 'Intra Median-Euc', 'Intra Mean-Cos']
    user_vals = [
        metrics['centroid_to_global']['user'],
        metrics['rmse_intra']['user'],
        metrics['median_euc_intra']['user'],
        metrics['mean_cos_intra']['user'],
    ]
    non_vals = [
        metrics['centroid_to_global']['non'],
        metrics['rmse_intra']['non'],
        metrics['median_euc_intra']['non'],
        metrics['mean_cos_intra']['non'],
    ]
    _write_tsv(
            "dist_metrics.txt",
            header=["metric", "user", "non"],
            rows=zip(names, user_vals, non_vals)
        )
    x = np.arange(len(names)); w = 0.38
    plt.figure(figsize=figsize)
    plt.bar(x - w/2, user_vals, width=w, label='User-related')
    plt.bar(x + w/2, non_vals,  width=w, label='Non-related')
    plt.xticks(x, names, rotation=20, ha='right')
    plt.ylabel('Value'); plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dist.png")

    plt.show()

    return {
        'metrics': metrics,
        'user_clusters': sorted(list(user_clusters)),
        'non_user_clusters': sorted(list(set(labels.astype(str)) - user_clusters)),
        'table': stat_table
    }


import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
def nearestN_utilization_expression_compare(
    adata_before,
    adata_after,
    cluster_col_before='orig_annot',
    cluster_col_after='cluster',
    user_col='user_selected',
    n_list=None,                 # 默认 50..1000 步长 50
    metric='cosine',
    extra_buffer=50,
    top_frac=0.30,               # ★ 这里生效：取“用户比例”最高的前 30% 簇
    figsize=(7,5),
    title='Nearest-N (expr): before vs after',
    debug_k: int = 50
):
    import numpy as np, pandas as pd
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt

    if n_list is None:
        n_list = list(range(50, 1001, 50))
    max_n = int(max(n_list))

    # ===== 数据与索引一致性 =====
    def _to_dense(X): return X.toarray() if hasattr(X, "toarray") else X
    X = _to_dense(adata_before.X)
    assert X.shape[0] == adata_after.n_obs and np.all(adata_before.obs_names == adata_after.obs_names), \
        "before/after 的 obs 顺序必须一致"

    labels_b = adata_before.obs[cluster_col_before].astype(str).values
    labels_a = adata_after.obs[cluster_col_after].astype(str).values

    def _ensure_bool(series):
        if series.dtype == bool: return series.values
        return series.astype(str).str.lower().isin({'true','1','yes','y','selected'}).values

    sel_bool = _ensure_bool(adata_before.obs[user_col])
    sel_idx = np.flatnonzero(sel_bool)
    assert sel_idx.size > 0, "没有用户选择细胞（user_selected=True）。"

    # ===== 用户相关簇（按“用户比例”选 top_frac）=====
    def get_user_clusters_by_proportion(labels, sel_idx, top_frac=0.30, min_clusters=1):
        lab = pd.Series(labels, name='cluster').astype(str)
        is_user = pd.Series(np.isin(np.arange(len(labels)), sel_idx), name='is_user')
        df = pd.concat([lab, is_user], axis=1)
        grp = df.groupby('cluster')
        cnt_all  = grp['is_user'].size().rename('n_all')
        cnt_user = grp['is_user'].sum().rename('n_user')
        frac = (cnt_user / cnt_all).fillna(0.0).rename('user_frac')

        # 选“比例最高的 top_frac”，同时过滤掉完全没有用户样本的簇（frac=0）
        k = max(min_clusters, int(np.ceil(len(frac) * float(top_frac))))
        # 次序：先按 frac 降序，再按 n_user 降序兜底
        order = frac.to_frame().join(cnt_user).sort_values(['user_frac','n_user'], ascending=False)
        keep = order[order['user_frac'] > 0.0].head(k).index.tolist()

        # 兜底：如果所有簇 user_frac 都是 0（极少见），退化为按 n_user 排序取 top k
        if len(keep) == 0:
            keep = cnt_user.sort_values(ascending=False).head(k).index.tolist()

        # 调试：打印前 10 个簇的比例与计数
        print("[User-clusters by proportion] top entries:")
        print(order.head(10).to_string())
        return set(map(str, keep))

    uc_b = get_user_clusters_by_proportion(labels_b, sel_idx, top_frac=top_frac)
    uc_a = get_user_clusters_by_proportion(labels_a, sel_idx, top_frac=top_frac)
    print(f"[Nearest-N] BEFORE user-clusters (top {int(top_frac*100)}% by proportion): {sorted(list(uc_b))}")
    print(f"[Nearest-N] AFTER  user-clusters (top {int(top_frac*100)}% by proportion): {sorted(list(uc_a))}")

    # ===== 最近邻（K 至少覆盖 max_n，并预留排除用户的 buffer）=====
    K_needed = max_n + sel_idx.size + extra_buffer
    K = min(X.shape[0], max(K_needed, int(debug_k)))
    if K < max_n:
        print(f"[Warn] K={K} < max_n={max_n}，n>{K} 的点将使用 n_eff=K。")

    nn = NearestNeighbors(n_neighbors=K, metric=metric).fit(X)
    query = X[sel_idx].mean(axis=0, keepdims=True)
    _, idx = nn.kneighbors(query)
    idx = idx[0]

    # ===== 调试：Top-K 的簇分布（两套标签）=====
    Kd = min(int(debug_k), idx.size)
    top_k = idx[:Kd]
    s_b = pd.Series(labels_b[top_k]).value_counts()
    s_a = pd.Series(labels_a[top_k]).value_counts()
    print(f"[Debug] Top-{Kd} neighbor clusters (BEFORE: {cluster_col_before}):\n{s_b.to_string()}")
    print(f"[Debug] Top-{Kd} neighbor clusters (AFTER : {cluster_col_after}):\n{s_a.to_string()}")
    head = min(20, Kd)
    print("[Debug] First neighbors (cell, before_label, after_label):")
    for j in range(head):
        print(f"  {adata_before.obs_names[top_k[j]]}  |  {labels_b[top_k[j]]}  |  {labels_a[top_k[j]]}")

    print_user_cluster_stats(labels_b, sel_idx, prefix="BEFORE")
    print_user_cluster_stats(labels_a, sel_idx, prefix="AFTER")

    # ===== 计算曲线 =====
    sel_set = set(sel_idx)
    def _curves(user_clusters, labels_for_membership):
        womask, mask = [], []
        for n in n_list:
            n_eff = min(n, len(idx))  # 防御：即使 K 不足
            # womask：直接前 n_eff
            topn = idx[:n_eff]
            womask.append(float(np.isin(labels_for_membership[topn], list(user_clusters)).mean()))
            # mask：按顺序取 n_eff 个非用户
            mask_list = []
            for j in idx:
                if j not in sel_set:
                    mask_list.append(j)
                    if len(mask_list) == n_eff:
                        break
            mask.append(float(np.isin(labels_for_membership[mask_list], list(user_clusters)).mean())
                        if len(mask_list) == n_eff else np.nan)
        return womask, mask

    wom_b, msk_b = _curves(uc_b, labels_b)
    wom_a, msk_a = _curves(uc_a, labels_a)

    _write_tsv(
            "nearestN.txt",
            header=["n", "before_womask", "before_mask", "after_womask", "after_mask"],
            rows=zip(n_list, wom_b, msk_b, wom_a, msk_a)
        )

    # ===== 画图 =====
    plt.figure(figsize=figsize)
    plt.plot(n_list, wom_b, marker='o', label=f'before-womask ({cluster_col_before})')
    plt.plot(n_list, msk_b, marker='s', label=f'before-mask   ({cluster_col_before})')
    plt.plot(n_list, wom_a, marker='^', label=f'after-womask  ({cluster_col_after})')
    plt.plot(n_list, msk_a, marker='x', label=f'after-mask    ({cluster_col_after})')
    plt.xlabel('N (neighbors in expression space)')
    plt.title(title) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("nearestN.png")  
    plt.show()


    return {
        'n_list': n_list,
        'before': {'womask': wom_b, 'mask': msk_b, 'user_clusters': sorted(list(uc_b))},
        'after' : {'womask': wom_a, 'mask': msk_a, 'user_clusters': sorted(list(uc_a))},
        'debug': {
            'K': int(K),
            'top_k_indices': top_k.tolist(),
            'top_k_before_counts': s_b.to_dict(),
            'top_k_after_counts':  s_a.to_dict(),
        }
    }


def print_user_cluster_stats(labels, sel_idx, prefix=""):
    """
    打印用户选择细胞在各簇中的数量和占比
    labels: ndarray / list, 每个细胞的簇标签
    sel_idx: ndarray / list, 用户选择的细胞索引
    prefix: str, 打印前缀（比如 'BEFORE' / 'AFTER'）
    """
    import pandas as pd
    labels = pd.Series(labels).astype(str)

    # 总数
    total_per_cluster = labels.value_counts().sort_index()

    # 用户选中的
    sel_labels = labels.iloc[sel_idx]
    user_per_cluster = sel_labels.value_counts().reindex(total_per_cluster.index, fill_value=0)

    # 占比
    frac = (user_per_cluster / total_per_cluster).fillna(0.0)

    print(f"\n[{prefix}] 用户选择细胞在各簇里的统计：")
    print("cluster | user_count | total_count | user_frac")
    for cl in total_per_cluster.index:
        print(f"{cl:>7} | {user_per_cluster[cl]:>10} | {total_per_cluster[cl]:>11} | {frac[cl]:.3f}")
