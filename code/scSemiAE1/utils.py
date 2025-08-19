import numpy as np
import torch

def extract_cells_by_index(adata, cell_idx_list, layer=None, device="cpu"):
    """
    从 AnnData 按位置索引提取细胞，返回表达矩阵的 torch.Tensor 和对应的细胞名字列表。
    细胞名字会被唯一化（以免重复），但不放入 tensor 中。

    Parameters:
        adata: AnnData object
        cell_idx_list: list of integer positional indices (0-based)
        layer: None 或 layer 名称（用哪个数据层，默认 adata.X）
        device: "cpu" 或 "cuda"

    Returns:
        cell_names: list of str, 选中细胞的名字（顺序对应 tensor 的行）
        tensor: torch.FloatTensor of shape (n_selected_cells, n_genes)
    """
    # 唯一化名字（防 warning 影响理解）
    adata.obs_names_make_unique()

    n_cells = adata.n_obs
    valid_idxs = []
    for i in cell_idx_list:
        if not isinstance(i, (int, np.integer)):
            continue
        if 0 <= i < n_cells:
            valid_idxs.append(int(i))
        else:
            print(f"Warning: index {i} out of bounds (0..{n_cells-1}), skipping.")

    if len(valid_idxs) == 0:
        raise ValueError("No valid cell indices provided.")

    # 去重但保留首次出现顺序
    seen = set()
    filtered_idxs = []
    for i in valid_idxs:
        if i not in seen:
            seen.add(i)
            filtered_idxs.append(i)

    # 取细胞名字（顺序对齐）
    cell_names = adata.obs_names[filtered_idxs].tolist()

    # 取表达矩阵
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        mat = adata.layers[layer][filtered_idxs]
    else:
        mat = adata.X[filtered_idxs]

    # 稀疏转 dense
    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    arr = np.array(mat, dtype=float)  # 保证是 float

    tensor = torch.from_numpy(arr).float().to(device)

    return cell_names, tensor



import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

def cluster_and_plot(
    adata,
    embeddings,                     # numpy array 或 pd.DataFrame（index=cell names）
    method: str = "leiden",
    resolution: float = 1.0,
    title_suffix: str = "",
    user_col: str = "user_selected",
    n_neighbors: int = 30,
    metric: str = "euclidean",
    savepath: str | None = None,
):
    """
    将 embeddings 写入 adata.obsm['X_scSemi']，基于 latent 重算邻居/UMAP/聚类。
    返回带有 UMAP/cluster 的 adata（保留原有 obs 列）。
    """
    A = adata.copy()

    # 1) 对齐顺序，并写入 obsm
    if isinstance(embeddings, pd.DataFrame):
        missing = set(A.obs_names) - set(embeddings.index)
        if missing:
            raise ValueError(f"embeddings 缺少 {len(missing)} 个细胞，例如: {list(missing)[:5]}")
        emb = embeddings.loc[A.obs_names].values
    else:
        emb = np.asarray(embeddings)
        if emb.shape[0] != A.n_obs:
            raise ValueError(f"embeddings 行数({emb.shape[0]}) != adata.n_obs({A.n_obs})")
    A.obsm["X_scSemi"] = emb

    # 2) 基于 latent 重建图与 UMAP
    sc.pp.neighbors(A, n_neighbors=n_neighbors, use_rep="X_scSemi", metric=metric)
    sc.tl.umap(A)

    # 3) 聚类
    if method.lower() == "leiden":
        sc.tl.leiden(A, resolution=resolution, key_added="cluster")
        cluster_key = "cluster"
    elif method.lower() == "louvain":
        sc.tl.louvain(A, resolution=resolution, key_added="cluster")
        cluster_key = "cluster"
    elif method.lower() == "kmeans":
        from sklearn.cluster import KMeans
        n_clusters = len(set(A.obs.get("true_label", []))) if "true_label" in A.obs else 10
        A.obs["cluster"] = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(emb).astype(str)
        cluster_key = "cluster"
    else:
        raise ValueError(f"unknown method {method}")

    # 4) 画图（确保用的是刚算的 UMAP）
    sc.pl.umap(A, color=[cluster_key, user_col] if user_col in A.obs else [cluster_key],
               size=10, wspace=0.4, show=False, title=f"UMAP ({method}) {title_suffix}")
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"[Saved] {savepath}")
    plt.show()

    return A


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

def _ensure_umap(adata, use_rep=None, n_neighbors=30, min_dist=0.5,
                 random_state=0, metric='euclidean', force=False):
    """
    若 force=True，必重算 neighbors + UMAP；否则仅当 'X_umap' 不在 obsm 时才计算。
    注意：neighbors() 没有 random_state 参数，random_state 传给 UMAP。
    """
    needs = force or ('X_umap' not in adata.obsm)
    if needs:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, metric=metric)
        sc.tl.umap(adata, random_state=random_state, min_dist=min_dist)

def _as_categorical_bool(series, true_label='selected', false_label='other'):
    """
    把任意类型的选择列稳健转换为两类分类：other / selected。
    仅 'true','1','yes','y','selected'（大小写不敏感）视为 True。
    """
    s = pd.Series(series.values, index=series.index)
    if s.dtype == bool:
        b = s
    else:
        b = s.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y', 'selected'})
    cat = pd.Categorical(
        np.where(b, true_label, false_label),
        categories=[false_label, true_label], ordered=True
    )
    return pd.Series(cat, index=series.index)

def umap_sixpanel_before_after(
    adata_before,
    adata_after,
    orig_label_col: str,              # before 里的原有注释列名（例如 'orig_annot'）
    new_cluster_col: str = 'cluster', # after 里的新聚类列名
    user_col: str = 'user_selected',
    use_rep_before=None,              # before 若无 UMAP，用哪个表示建图（None=adata.X）
    use_rep_after='X_scSemi',         # after 若无 UMAP，用哪个表征建图（默认 latent）
    n_neighbors: int = 30,
    min_dist: float = 0.5,
    metric: str = 'euclidean',
    random_state: int = 0,
    point_size: int = 8,
    figsize=(18, 10),
    savepath: str | None = None
):
    """
    2×3 六联图：
      [0,0] Before: original annotation（原注释）
      [0,1] Before: user-selected cells（原表示上的用户掩码）
      [0,2] Before: new clustering (projected)（把 after 的聚类投影回 before）
      [1,0] After : new clustering（latent 上的新聚类）
      [1,1] After : original annotation（把原注释带到 after 只作对比）
      [1,2] After : user-selected cells（latent 上的用户掩码）
    """
    # 0) 保护：不在函数内修改来者对象的关键列
    A0 = adata_before
    A1 = adata_after

    # 1) 确保两边有 UMAP（before 可复用，after 强制基于指定表征重算）
    _ensure_umap(A0, use_rep=use_rep_before, n_neighbors=n_neighbors,
                 min_dist=min_dist, metric=metric, random_state=random_state, force=False)
    _ensure_umap(A1, use_rep=use_rep_after,  n_neighbors=n_neighbors,
                 min_dist=min_dist, metric=metric, random_state=random_state, force=True)

    # 2) 同步列（按 obs_names 对齐，不覆盖原列）
    # 2.1 把 after 的聚类结果投影到 before
    new_on_before = f'{new_cluster_col}__on_before'
    A0.obs[new_on_before] = (
        pd.Series(A1.obs[new_cluster_col].astype(str).values, index=A1.obs_names)
          .reindex(A0.obs_names)
          .fillna('NA')
          .values
    )
    # 2.2 把 before 的原注释列投影到 after（仅用于可视化对比）
    orig_on_after = f'{orig_label_col}__on_after'
    if orig_label_col not in A1.obs.columns:
        A1.obs[orig_on_after] = (
            pd.Series(A0.obs[orig_label_col].astype(str).values, index=A0.obs_names)
              .reindex(A1.obs_names)
              .fillna('NA')
              .values
        )
        show_orig_on_after = orig_on_after
    else:
        show_orig_on_after = orig_label_col

    # 3) 用户列规范化（保持两边都有且是二类分类）
    for A in (A0, A1):
        if user_col not in A.obs.columns:
            A.obs[user_col] = False
        A.obs[user_col] = _as_categorical_bool(A.obs[user_col])

    # 4) 画六联图
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    sc.pl.umap(
        A0, color=[orig_label_col], ax=axes[0, 0], show=False,
        title='Before: original annotation', legend_loc='on data', size=point_size
    )
    sc.pl.umap(
        A0, color=[user_col], ax=axes[0, 1], show=False,
        title='Before: user-selected cells',
        palette={'other': '#D3D3D3', 'selected': '#D62728'},
        legend_loc=None, size=point_size
    )
    sc.pl.umap(
        A0, color=[new_on_before], ax=axes[0, 2], show=False,
        title='Before: new clustering (projected)', legend_loc='on data', size=point_size
    )
    sc.pl.umap(
        A1, color=[new_cluster_col], ax=axes[1, 0], show=False,
        title='After: new clustering', legend_loc='on data', size=point_size
    )
    sc.pl.umap(
        A1, color=[show_orig_on_after], ax=axes[1, 1], show=False,
        title='After: original annotation', legend_loc='on data', size=point_size
    )
    sc.pl.umap(
        A1, color=[user_col], ax=axes[1, 2], show=False,
        title='After: user-selected cells',
        palette={'other': '#D3D3D3', 'selected': '#D62728'},
        legend_loc=None, size=point_size
    )

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"[Saved] {savepath}")
    plt.savefig("umap_compare.png")
    plt.show()

    return fig, axes
import numpy as np
import numpy as np

def to_dense_ndarray(X):
    """把可能的稀疏/torch/列表统一成 2D numpy.ndarray(float32)。"""
    if hasattr(X, "toarray"):          # scipy.sparse
        X = X.toarray()
    elif not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X

def prepare_labeled_subset(
    adata,
    annot_col="orig_annot",
    user_col="user_selected",
    user_class_name="USER_CLASS",
    # —— 过滤 unknown 的设置 —— 
    unknown_values=("unknown", "na", "nan", "unassigned", "None", "NONE", "Unassigned"),
    # —— 抽样设置（非用户样本）——
    keep_ratio=None,          # 每个真实类别按比例抽样，如 0.2；与 per_class_cap 二选一
    per_class_cap=None,       # 每个真实类别最多保留多少个，如 10；与 keep_ratio 二选一
    min_per_class=1,          # 每类至少保留多少个（在有样本的前提下）
    seed=0,
    encode_labels=True,
):
    """
    返回：用于监督训练的子集（X_l, ids_l, y_l），并附带若干辅助信息（字典）。
    规则：
      - 用户选择的样本：全部保留，标签强制为 user_class_name。
      - 非用户样本：若标签在 unknown_values 或为缺失，则剔除；剩余按 keep_ratio/per_class_cap 分层抽样。
    """
    rng = np.random.RandomState(seed)

    # 1) 特征矩阵（所有样本）
    X_all = to_dense_ndarray(adata.X)
    N = X_all.shape[0]
    obs_names = adata.obs_names.to_numpy()

    # 2) 原标签（字符串）
    raw_labels = adata.obs[annot_col].astype(str).to_numpy()

    # 3) 用户布尔
    if adata.obs[user_col].dtype == bool:
        is_user = adata.obs[user_col].to_numpy()
    else:
        is_user = adata.obs[user_col].astype(str).str.lower().isin(
            {"true","1","yes","y","selected"}
        ).to_numpy()

    # 4) unknown 的定义（统一小写比较）
    unknown_set = {str(u).strip().lower() for u in unknown_values}

    # 5) 先把用户样本挑出来（它们的标签强制为 user_class_name）
    user_idx = np.flatnonzero(is_user)
    user_labels = np.array([user_class_name] * user_idx.size, dtype=object)

    # 6) 非用户样本里，过滤 unknown/缺失
    non_user_idx_all = np.flatnonzero(~is_user)
    raw_non_user_lab = raw_labels[non_user_idx_all].astype(str)
    m_known = np.array([lab.strip().lower() not in unknown_set for lab in raw_non_user_lab], dtype=bool)
    non_user_idx = non_user_idx_all[m_known]
    non_user_lab = raw_labels[non_user_idx].astype(str)

    # 7) 在非用户可用样本中，做“按类分层抽样”
    #    两种方式：keep_ratio 或 per_class_cap（传一个即可）
    if (keep_ratio is None) and (per_class_cap is None):
        # 若两者都未指定，默认“不过度抽样”：每类全保留（可自行改为更小的监督集）
        chosen_idx = non_user_idx.copy()
    else:
        chosen = []
        uniq = np.unique(non_user_lab)
        for c in uniq:
            pool = non_user_idx[ non_user_lab == c ]
            if pool.size == 0:
                continue
            if keep_ratio is not None:
                k = max(min_per_class, int(np.ceil(pool.size * float(keep_ratio))))
            else:
                k = max(min_per_class, int(per_class_cap))
            k = min(k, pool.size)
            if k > 0:
                chosen.append(rng.choice(pool, size=k, replace=False))
        chosen_idx = np.concatenate(chosen, axis=0) if len(chosen) > 0 else np.array([], dtype=int)

    chosen_lab = raw_labels[chosen_idx].astype(str)

    # 8) 监督训练用的“最终有标签子集” = 用户样本 ∪ 被抽样的非用户样本
    labeled_idx = np.concatenate([user_idx, chosen_idx], axis=0)
    labeled_lab = np.concatenate([user_labels, chosen_lab], axis=0)

    # 9) 编码标签（可选）
    if encode_labels:
        uniq = np.unique(labeled_lab)
        cls2idx = {c: i for i, c in enumerate(uniq)}
        y_l = np.array([cls2idx[c] for c in labeled_lab], dtype=np.int64)
    else:
        cls2idx = None
        y_l = labeled_lab.tolist()

    # 10) 整理输出
    X_l = X_all[labeled_idx]
    ids_l = obs_names[labeled_idx]

    # 额外信息（方便你打印/检查）
    info = {
        "num_total": int(N),
        "num_user": int(user_idx.size),
        "num_non_user_known": int(non_user_idx.size),
        "num_labeled_final": int(labeled_idx.size),
        "per_class_counts_after_sampling": {},
        "mapping_cls2idx": cls2idx,
        "indices": {
            "user_idx": user_idx.tolist(),
            "non_user_idx_kept_pool": non_user_idx.tolist(),
            "non_user_idx_chosen": chosen_idx.tolist(),
            "labeled_idx": labeled_idx.tolist(),
        },
    }
    # 统计分布（字符串标签视角）
    _, cnts = np.unique(labeled_lab, return_counts=True)
    for k, v in zip(np.unique(labeled_lab), cnts):
        info["per_class_counts_after_sampling"][str(k)] = int(v)

    return X_l, ids_l.tolist(), y_l, info
