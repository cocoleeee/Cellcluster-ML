import argparse
parser = argparse.ArgumentParser(description="Training scSemiAE")
# Training settings:
parser.add_argument("-dpath", "--data_path", default="./dataset/", help="path to the dataset folder")
parser.add_argument("-spath", "--save_path", default="./output/", help="path to output directory")

parser.add_argument("-lsize", "--lab_size", type=int, default=10, help="labeled set size for each cell type (default: 10)")
parser.add_argument("-lratio", "--lab_ratio", type=float, default=-1, help="labeled set ratio for each cell type (default: -1)")

parser.add_argument("-s", "--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('-pretrain_batch', '--pretrain_batch', type=int, default=100)
parser.add_argument('-nepoch', '--epochs', type=int, default=60)
parser.add_argument('-nepoch_pretrain', '--epochs_pretrain', type=int, default=50)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-lrS', '--lr_scheduler_step', type=int, default=10)
parser.add_argument('-lrG', '--lr_scheduler_gamma', type=float, default=0.5)
parser.add_argument('-lbd', '--Lambda', type=float, default=1)
parser.add_argument('-v', '--visual', type=bool, default=False)
# --- 可选开关 ---
parser.add_argument('--use_gene_weighting', action='store_true', help='启用特征重加权（推荐）')
parser.add_argument('--pseudo_M', type=int, default=500, help='伪正扩充数量上限（默认 500）')
parser.add_argument('--neg_ratio', type=float, default=2.0, help='负样本与正样本数量比例（默认 2x）')
args = parser.parse_args()


COL_NAME="cluster-10"

# make sure saving path exists
import os
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# ================ Imports ================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib.pyplot as plt
import scanpy as sc
import anndata

from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances

from model.scSemiAE import scSemiAE
from model.dataset import ExperimentDataset
from model.inference import knn_infer,  louvain_cluster_infer, kmeans_cluster_infer
from model.metrics import compute_scores, compute_scores_for_cls

# 你自己的工具
from utils import extract_cells_by_index, cluster_and_plot, umap_sixpanel_before_after
# from eval import plot_similar_n, compute_dist_metrics, nearestN_utilization_expression
from eval import plot_similar_n_compare, compute_dist_metrics_compare,nearestN_utilization_expression_compare
    
from utils import prepare_labeled_subset,to_dense_ndarray

# ================ Utils ================
def set_all_seeds(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让结果更可复现（牺牲一点速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_all_seeds(args.seed)

# 主函数入口
def main():
     # ============ 设备 & 随机种子 ============
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: 有可用 CUDA 设备，建议使用 --cuda 启动")
    device = 'cuda:0' if (torch.cuda.is_available() and args.cuda) else 'cpu'
    args.device = device

    # ============ 读数据（原始副本） ============
    adata_raw = sc.read_h5ad("sc_sampled.h5ad")
    adata_raw.obs_names_make_unique()
    X = adata_raw.X.toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X  # (N, G)
    cell_id = adata_raw.obs_names.tolist()

    # ============ 用户选择（严格布尔 & 仅一次） ============

    cell_idx_list=[12,98,114,147,151,218,219,228,234,236,240,295,318,364,409,443,525,536,572,627,647,676,779,783,811,836,892,894,904,930,979,983,999,1004,1031,1038,1078,1090,1154,1168,1185,1194,1197,1213,1248,1295,1326,1333,1365,1367,1373,1393,1429,1448,1453,1529,1565,1583,1616,1625,1632,1670,1686,1702,1724,1748,1780,1827,1839,1854,1903,1906,1936,1944,1977,1978,1986,1999,2005,2021,2024,2084,2098,2100,2190,2205,2239,2247,2262,2294,2304,2336,2355,2446,2448,2467,2478,2488,2491,2530,2536,2538,2542,2545,2546,2560,2606,2672,2698,2702,2788,2801,2807,2824,2825,2839,2840,2878,2884,2909,2924,2967,2986,97,645,681,695,832,1178,1376,1400,1770,1976,2311,2664,2707,2855]

    pos_idx = np.array(cell_idx_list, dtype=int)
    pos_idx = pos_idx[(pos_idx >= 0) & (pos_idx < adata_raw.n_obs)]
    pos_names = adata_raw.obs_names[pos_idx].tolist()

    user_sel = np.zeros(adata_raw.n_obs, dtype=bool)
    user_sel[pos_idx] = True
    adata_raw.obs['user_selected'] = user_sel  # bool！

    # ==== 构造 prior_label：把原注释 + USER 合在一套标签空间里 ====
    import pandas as pd
    lab_col = 'leiden-1'                     # 原始注释列；如果叫别的，改成对应列名
    lab = adata_raw.obs[lab_col].astype('category').astype(str)
    lab[adata_raw.obs['user_selected'].astype(bool)] = 'USER'  # 用户细胞统一成特殊类

    cats = pd.Index(sorted(lab.unique()))
    cat2id = {c:i for i,c in enumerate(cats)}                  # str 类名 -> int
    prior_id = lab.map(cat2id).astype(int)                     # 与 obs 顺序对齐

    # 做一个 cell_id -> prior_id 的字典，未找到时默认 -1（即“无先验”）
    prior_label_dict = {cell: int(pid) for cell, pid in zip(adata_raw.obs_names, prior_id)}


    # ============ 构建 pretrain / labeled（先不启用伪正/重加权） ============
    Xw = X
    data_tensor = torch.tensor(Xw, dtype=torch.float)
    pretrain_data = ExperimentDataset(data_tensor, cell_id, [])  # 预训练不需要标签

    # 负样本：非用户中随机取 max(正样本*ratio, 50)
    all_idx = np.arange(adata_raw.n_obs)
    neg_pool = np.setdiff1d(all_idx, pos_idx)
    neg_n = min(len(neg_pool), max(int(len(pos_idx) * args.neg_ratio), 50))
    rng = np.random.default_rng(args.seed)
    neg_idx = rng.choice(neg_pool, size=neg_n, replace=False)

    def _tensor_by_idx(idxs):
        arr = Xw[idxs]
        return torch.tensor(arr, dtype=torch.float, device='cpu')

    pos_tensor = _tensor_by_idx(pos_idx)
    neg_tensor = _tensor_by_idx(neg_idx)

    lab_tensor = torch.cat([pos_tensor, neg_tensor], dim=0)
    lab_names  = adata_raw.obs_names[pos_idx].tolist() + adata_raw.obs_names[neg_idx].tolist()  # 字符串 ID
    lab_labels = [1]*len(pos_idx) + [0]*len(neg_idx)



    labeled_data = ExperimentDataset(lab_tensor, lab_names, lab_labels)

    # from utils import prepare_labeled_dataset,to_dense_ndarray

#     labeled_data, labeled_cellid, labeled_lab = prepare_labeled_dataset(
#     adata_raw,
#     annot_col="leiden-1",
#     user_col="user_selected",
#     user_class_name="USER_CLASS"
# )
    

    # ===== labeled_data =====
    X_l, ids_l, y_l, info = prepare_labeled_subset(
        adata_raw,
        annot_col="annotation",
        user_col="user_selected",
        user_class_name="USER_CLASS",
        unknown_values=("unknown"),
        keep_ratio=0.2,          
        min_per_class=1,
        seed=args.seed,
        encode_labels=True
    )


    labeled_data = ExperimentDataset(X_l, ids_l, y_l)  # 直接塞入就行


    # labeled_data = ExperimentDataset(labeled_data, labeled_cellid, labeled_lab)

    # ============ 训练 ============
    model = scSemiAE(args, labeled_data, pretrain_data, hid_dim_1=500, hid_dim_2=50)
    model.user_set = set(adata_raw.obs_names[pos_idx].tolist())  # 与 Dataset 的 cells 同为字符串





    # 简短命中检查
    for b, (_, _, cells) in enumerate(model.train_loader):
        hits = len(set(cells) & model.user_set)
        print(f"[debug] warmup check - batch{b}: user hits = {hits}/{len(cells)}")
        if b >= 2: break

    embeddings_df = model.train()  # pd.DataFrame(index=cells)

    # ============ 对齐 latent & 构建 before/after 两份对象 ============
    # 对齐顺序（关键！）
    missing = set(adata_raw.obs_names) - set(embeddings_df.index)
    if missing:
        raise ValueError(f"embeddings 缺少 {len(missing)} 个细胞样本，例如: {list(missing)[:5]}")
    embeddings_df = embeddings_df.loc[adata_raw.obs_names]
    emb = embeddings_df.values

    # AFTER：基于 latent 的新聚类
    adata_after = adata_raw.copy()
    adata_after.obsm['X_scSemi'] = emb
    # sc.pp.neighbors(adata_after, n_neighbors=50, use_rep='X_scSemi', metric='euclidean')
    sc.pp.neighbors(adata_after, n_neighbors=50, use_rep='X_scSemi', metric='cosine') 
    sc.tl.umap(adata_after, random_state=args.seed)
    sc.tl.leiden(adata_after, key_added=COL_NAME, resolution=1.0)

    # ========= BEFORE：沿用原始 UMAP，不重算 =========
    adata_before = adata_raw.copy()

    # 如果原文件里已经有 UMAP 坐标（通常是 'X_umap'），我们就直接用来画图
    if 'X_umap' not in adata_before.obsm:
        # 若没有，就只在“真的缺失”的情况下再算一次（避免覆盖）
        sc.pp.neighbors(adata_before, n_neighbors=30, use_rep=None, metric='euclidean')
    sc.tl.umap(adata_before, random_state=args.seed)


    ORIG_LABEL = 'leiden-1'
    if ORIG_LABEL not in adata_before.obs.columns:
        # 想要一个“原结果标签”用于上图，但不改变 UMAP 坐标
        # 只算邻居+leiden，不再调用 sc.tl.umap，UMAP 坐标保持不变
        sc.pp.neighbors(adata_before, n_neighbors=30, use_rep=None, metric='euclidean')
        sc.tl.leiden(adata_before, key_added='orig_annot', resolution=1.0)
        ORIG_LABEL = 'orig_annot'  # 用刚算的标签在图上着色

    # ============ 六联图（after 必须用 latent） ============
    save_png = os.path.join(args.save_path, 'umap_before_after_six.png')
    umap_sixpanel_before_after(
        adata_before=adata_before,
        adata_after=adata_after,
        orig_label_col='leiden-1',
        new_cluster_col=COL_NAME,
        user_col='user_selected',
        use_rep_before=None,
        use_rep_after='X_scSemi',
        n_neighbors=30,
        min_dist=0.5,
        metric='euclidean',
        random_state=args.seed,
        point_size=6,
        figsize=(18, 10),
        savepath=save_png
    )
    print("[Saved]", save_png)

    # —— 评估前：把类别型的 user_selected → 布尔列 user_selected_bool
    adata_after.obs['user_selected_bool'] = (
        adata_after.obs['user_selected'].astype(str).str.lower().eq('selected')
    )
    

        # 三项评估改成用布尔列
    # Similar-N 对比
    _ = plot_similar_n_compare(
        adata_before, adata_after,
        cluster_col_before='leiden-1',   # 你 before 的列名
        cluster_col_after=COL_NAME,       # 你 after 的列名
        max_n=20, metric='cosine'
    )

    # Dist 指标对比
    res = compute_dist_metrics_compare(
        adata_after,
        cluster_col=COL_NAME,
        user_col='user_selected',
        top_frac=0.30,
        title='After | User vs Non (UMAP)'
    )

    # Nearest-N（表达空间）对比
    res = nearestN_utilization_expression_compare(
        adata_before,
        adata_after,
        cluster_col_before="leiden-1",   # before 的聚类列
        cluster_col_after=COL_NAME,       # after 的聚类列
        user_col="user_selected",          # 用户标记列
        n_list=list(range(50, 1001, 50)),   # 最近邻数量 N 序列
        metric="cosine",                   # 距离度量
        debug_k=50                         # 打印前 50 个近邻的分布
    )





 

    # ============ 保存 latent ============
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        embd_save_path = os.path.join(args.save_path, "scSemiAE_embeddings.csv")
        embeddings_df.to_csv(embd_save_path)
        print("[Saved]", embd_save_path)


    # adata_after.write_h5ad("sc_sampled.h5ad", compression="gzip")
    # 
    print("######################## over!")
    # print("######################## over!")



# 程序入口
if __name__ == "__main__":
    main()
