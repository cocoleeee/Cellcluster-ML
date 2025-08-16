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
    # 你已有的索引列表
    cell_idx_list = [7,10,25,37,47,61,72,81,93,103,107,119,120,126,127,144,171,200,213,220,251,254,257,258,260,271,275,276,282,293,305,308,309,312,346,376,378,379,425,427,437,438,452,460,469,494,497,501,516,535,538,540,559,563,578,584,595,601,605,614,632,637,650,652,656,661,662,665,685,707,711,730,733,738,742,743,772,773,775,776,778,784,804,810,812,839,840,851,854,860,868,869,884,896,905,908,912,920,921,941,949,965,967,970,971,978,981,1003,1013,1028,1047,1049,1052,1060,1077,1083,1087,1089,1092,1112,1113,1121,1125,1136,1148,1151,1162,1163,1165,1173,1181,1182,1192,1204,1208,1216,1217,1241,1258,1263,1268,1282,1294,1322,1336,1338,1341,1349,1350,1361,1364,1371,1372,1382,1384,1394,1407,1410,1417,1421,1436,1441,1464,1467,1473,1474,1475,1490,1509,1515,1517,1520,1527,1545,1546,1549,1567,1578,1581,1584,1587,1595,1603,1606,1607,1617,1619,1620,1621,1627,1630,1637,1642,1656,1672,1673,1678,1679,1680,1689,1710,1718,1728,1732,1733,1739,1751,1765,1769,1773,1781,1790,1801,1804,1816,1822,1828,1830,1835,1840,1850,1856,1858,1898,1911,1924,1937,1947,1948,1949,1961,1966,1970,1972,1973,1974,1975,1981,1983,1994,1997,2000,2001,2010,2014,2022,2023,2034,2046,2057,2064,2070,2075,2080,2085,2106,2133,2145,2150,2155,2174,2185,2189,2195,2196,2207,2213,2226,2229,2231,2249,2253,2255,2256,2271,2273,2277,2283,2285,2286,2290,2291,2295,2303,2312,2313,2317,2320,2323,2325,2327,2347,2351,2366,2376,2377,2384,2395,2397,2401,2403,2405,2406,2417,2421,2429,2441,2456,2464,2469,2472,2481,2483,2489,2497,2500,2503,2504,2521,2522,2525,2535,2539,2544,2548,2553,2556,2571,2575,2582,2583,2591,2615,2616,2617,2621,2623,2626,2629,2641,2648,2660,2661,2674,2676,2682,2688,2694,2696,2703,2705,2716,2721,2724,2727,2732,2734,2740,2743,2752,2761,2762,2765,2766,2776,2798,2802,2822,2828,2834,2838,2849,2870,2881,2888,2893,2911,2916,2917,2925,2926,2931,2936,2937,2942,2950,2951,2954,2961,2972,2976,2978,2981,2997,612,902,0,3,12,98,114,128,147,151,185,218,219,228,234,236,240,241,287,295,304,318,364,409,443,445,453,480,519,525,536,562,572,627,647,676,694,698,705,724,749,779,783,787,811,892,893,894,904,917,930,974,979,983,986,999,1004,1031,1038,1078,1081,1090,1100,1130,1168,1180,1185,1194,1197,1248,1256,1278,1326,1333,1340,1346,1347,1356,1365,1373,1389,1409,1429,1448,1449,1453,1476,1477,1479,1514,1565,1583,1616,1625,1632,1670,1676,1686,1702,1724,1741,1748,1749,1796,1823,1827,1839,1848,1854,1903,1906,1952,1969,1978,1986,1999,2005,2019,2021,2024,2084,2098,2099,2100,2190,2205,2216,2239,2247,2262,2294,2304,2322,2332,2336,2355,2413,2446,2448,2467,2478,2488,2491,2530,2536,2538,2545,2546,2560,2563,2665,2672,2698,2702,2777,2779,2788,2793,2801,2807,2812,2824,2825,2839,2840,2865,2878,2909,2924]

    pos_idx = np.array(cell_idx_list, dtype=int)
    pos_idx = pos_idx[(pos_idx >= 0) & (pos_idx < adata_raw.n_obs)]
    pos_names = adata_raw.obs_names[pos_idx].tolist()

    user_sel = np.zeros(adata_raw.n_obs, dtype=bool)
    user_sel[pos_idx] = True
    adata_raw.obs['user_selected'] = user_sel  # bool！

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

    # ============ 训练 ============
    model = scSemiAE(args, labeled_data, pretrain_data, hid_dim_1=500, hid_dim_2=50)
    model.user_set = set(adata_raw.obs_names[pos_idx].tolist())  # 与 Dataset 的 cells 同为字符串

    # （可选）增强个性化权重做“能量测试”
    model.alpha_user = 12.0
    model.beta_user  = 10.0
    model.use_triplet = True
    model.w_triplet   = 0.8
    model.triplet_margin = 0.6
    model.use_supcon  = True
    model.w_supcon    = 0.8
    model.supcon_tau  = 0.07

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
    sc.pp.neighbors(adata_after, n_neighbors=30, use_rep='X_scSemi', metric='euclidean')
    sc.tl.umap(adata_after, random_state=args.seed)
    sc.tl.leiden(adata_after, key_added='cluster', resolution=1.0)

    # ========= BEFORE：沿用原始 UMAP，不重算 =========
    adata_before = adata_raw.copy()

    # 如果原文件里已经有 UMAP 坐标（通常是 'X_umap'），我们就直接用来画图
    if 'X_umap' not in adata_before.obsm:
        # 若没有，就只在“真的缺失”的情况下再算一次（避免覆盖）
        sc.pp.neighbors(adata_before, n_neighbors=30, use_rep=None, metric='euclidean')
    sc.tl.umap(adata_before, random_state=args.seed)

# 原注释列名（你文件里叫什么就用什么；若已有如 'leiden-1'、'cell_type'，直接用）
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
        new_cluster_col='cluster',
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
        cluster_col_after='cluster',       # 你 after 的列名
        max_n=20, metric='cosine'
    )

    # Dist 指标对比
    _ = compute_dist_metrics_compare(
        adata_before, adata_after,
        cluster_col_before='leiden-1',
        cluster_col_after='cluster',
        user_col='user_selected',          # 类别/字符串/布尔都可
        top_frac=0.30
    )

    # Nearest-N（表达空间）对比
    _ = nearestN_utilization_expression_compare(
        adata_before, adata_after,
        cluster_col_before='leiden-1',
        cluster_col_after='cluster',
        user_col='user_selected',
        metric='cosine', top_frac=0.30
    )




    # # ============ 评估（在 after 上评新聚类） ============
    # from eval import plot_similar_n, compute_dist_metrics, nearestN_utilization_expression
    # _ = plot_similar_n(adata_after, max_n=20)
    # _ = compute_dist_metrics(adata_after, plot=True)
    # _ = nearestN_utilization_expression(
    #     adata=adata_after,
    #     cluster_col="cluster",
    #     user_col="user_selected",
    #     metric="cosine",
    #     top_frac=0.30,
    #     plot=True
    # )

    # ============ 保存 latent ============
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        embd_save_path = os.path.join(args.save_path, "scSemiAE_embeddings.csv")
        embeddings_df.to_csv(embd_save_path)
        print("[Saved]", embd_save_path)

    print("######################## over!")
    # print("######################## over!")

# 程序入口
if __name__ == "__main__":
    main()
