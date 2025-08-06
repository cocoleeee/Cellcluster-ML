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




import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt

def cluster_and_plot(adata, embeddings, method="louvain", resolution=1.0, title_suffix=""):
    """
    adata: 原始 AnnData（需要和 embeddings 样本顺序一致）
    embeddings: numpy array of shape (n_obs, embedding_dim) from scSemiAE
    method: "louvain" / "kmeans" / "knn"  （这里只实现 louvain 和 kmeans）
    resolution: 分辨率，用于 louvain
    """
    # 1. 构造一个新的 AnnData 用 embedding 作为表示
    em = anndata.AnnData(X=embeddings)
    em.obs_names = adata.obs_names  # 对齐细胞名

    # 2. 计算邻居图 & UMAP
    sc.pp.neighbors(em, n_neighbors=30, use_rep='X')  # 'X' 是 embeddings
    sc.tl.umap(em)

    # 3. 聚类
    if method.lower() == "louvain":
        sc.tl.louvain(em, resolution=resolution, key_added="louvain")  # 结果存 em.obs["louvain"]
        cluster_key = "louvain"
    elif method.lower() == "leiden":
        sc.tl.leiden(em, resolution=resolution, key_added="leiden")
        cluster_key = "leiden"
    elif method.lower() == "kmeans":
        from sklearn.cluster import KMeans
        # 估计类数为 embedding 上的已有 label 数（可替换成固定 n_clusters）
        n_clusters = len(set(adata.obs.get("true_label", []))) if "true_label" in adata.obs else 10
        km = KMeans(n_clusters=n_clusters, random_state=0)
        em.obs["kmeans"] = km.fit_predict(embeddings).astype(str)
        cluster_key = "kmeans"
    else:
        raise ValueError(f"unknown method {method}")
    
    em.obs['cluster'] = em.obs[cluster_key].astype(str).values

    # 4. 可视化 UMAP，按聚类上色
    sc.pl.umap(em,size=10, color=[cluster_key], title=f"UMAP {method} {title_suffix}", show=True)
    return em  # 返回带聚类和 UMAP 结果的 AnnData
