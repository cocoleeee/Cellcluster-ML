import argparse
parser = argparse.ArgumentParser(description="Training scSemiAE")
# Training settings:
parser.add_argument("-dpath", "--data_path", default="./dataset/", help="path to the dataset folder")
parser.add_argument("-spath", "--save_path", default="./output/", help="path to output directory")

parser.add_argument("-lsize", "--lab_size", type=int, default=10, help="labeled set size for each cell type (default: 10)")
parser.add_argument("-lratio", "--lab_ratio", type=float, default=-1, help="labeled set ratio for each cell type (default: -1)")

parser.add_argument("-s", "--seed", type=int, default=0, help="random seed for loading dataset (default: 0)")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('-pretrain_batch', '--pretrain_batch', type=int, help="Batch size for pretraining.Default:100", default=100)
parser.add_argument('-nepoch', '--epochs', type=int, help='number of epochs to train for', default=60)
parser.add_argument('-nepoch_pretrain', '--epochs_pretrain', type=int, help='number of epochs to pretrain for', default=50)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for the model, default=0.001', default=0.001)
parser.add_argument('-lrS', '--lr_scheduler_step', type=int, help='StepLR learning rate scheduler step, default=10', default=10)
parser.add_argument('-lrG', '--lr_scheduler_gamma', type=float, help='StepLR learning rate scheduler gamma, default=0.5', default=0.5)
parser.add_argument('-lbd', '--Lambda', type=float, help='weight for L2, default=1', default=1)
parser.add_argument('-v', '--visual', type=bool, help='visualization of data. default=False', default=False)
args = parser.parse_args()

# make sure saving path exists
import os
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

import torch
from model.scSemiAE import scSemiAE
from model.dataset import ExperimentDataset
from model.inference import knn_infer,  louvain_cluster_infer, kmeans_cluster_infer
from model.metrics import compute_scores, compute_scores_for_cls
import scanpy as sc
import anndata
import torch
import matplotlib.pyplot as plt
from data import Data
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# 主函数入口
def main():
    # 设置计算设备（GPU 优先）
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: 有可用 CUDA 设备，建议使用 --cuda 启动")
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    args.device = device  # 保存到 args 中方便传递

    # # 加载数据
    # if args.lab_ratio != -1:  # 如果设置了标签比例
    #     dataset = Data(args.data_path, labeled_ratio=args.lab_ratio, seed=args.seed)
    # else:  # 否则使用标签数量
    #     dataset = Data(args.data_path, labeled_size=args.lab_size, seed=args.seed)

    # # 获取数据与标签信息
    # data, lab_full, labeled_idx, unlabeled_idx, info = dataset.load_all()

    # # 转为 torch tensor 格式
    # data = torch.tensor(data, dtype=torch.float)
    # cell_id = info["cell_id"]

    # # 分别处理有标签和无标签数据
    # labeled_data = data[labeled_idx, :]

    # labeled_lab = lab_full[labeled_idx].tolist()
    # # print(labeled_lab)

    # # unlabeled_lab = lab_full[unlabeled_idx].tolist()
    # # print(unlabeled_lab)
    # unlabeled_lab=[]
    # labeled_cellid = cell_id[labeled_idx].tolist()




    # # 构建 PyTorch 数据集
    # pretrain_data = ExperimentDataset(data, cell_id, lab_full)
    # labeled_data = ExperimentDataset(labeled_data, labeled_cellid, labeled_lab)



    adata = sc.read_h5ad("sc_sampled.h5ad")
    data=adata.X.toarray()
    data = torch.tensor(data, dtype=torch.float)

    adata.obs_names_make_unique()
    cell_id = adata.obs_names.tolist()  # list of cell names
    lab_full=[]

    from utils import extract_cells_by_index
    cell_idx_list = [13,42,46,66,70,112,166,194,290,332,342,405,444,513,518,537,710,720,723,950,957,963,1084,1085,1111,1132,1160,1207,1252,1308,1311,1337,1354,1443,1491,1536,1547,1585,1703,1712,1750,1800,1908,1928,1962,2065,2112,2131,2166,2292,2305,2324,2343,2346,2350,2382,2440,2715,2745,2747,2768,2932,2944,2969,2977,2992]
    labeled_cellid, labeled_data = extract_cells_by_index(adata, cell_idx_list)
    # print(cell_id)
    # print(expr_df)
    # expr_df=torch.tensor(expr_df, dtype=torch.float)
    length=len(cell_idx_list)
    labeled_lab = [0] * length

    all_idx = list(range(adata.n_obs))
    unlabeled_idx = [i for i in all_idx if i not in all_idx]
    unlabeled_lab=[]







    pretrain_data = ExperimentDataset(data, cell_id, lab_full)
    labeled_data = ExperimentDataset(labeled_data, labeled_cellid, labeled_lab)








    # 初始化并训练模型
    model = scSemiAE(args, labeled_data, pretrain_data, hid_dim_1=500, hid_dim_2=50)
    embeddings = model.train()  # 获取嵌入表示

    # 保存嵌入向量为 CSV
    if args.save_path:
        embd_save_path = args.save_path + "scSemiAE_embeddings.csv"
        embeddings.to_csv(embd_save_path)

    # 转为 numpy 形式用于评估
    embeddings = embeddings.values

    # ======== 使用 KNN 推断未标记样本 ========
    # unlabeled_lab_knn_pred = knn_infer(embeddings, lab_full, labeled_idx, unlabeled_idx)
    # scores_knn = compute_scores(unlabeled_lab, unlabeled_lab_knn_pred)
    # print("KNN:")
    # print(scores_knn)

    # ======== 使用 Louvain 聚类评估聚类性能 ========
    # pred = louvain_cluster_infer(embeddings, unlabeled_idx)
    # scores_louvain = compute_scores_for_cls(unlabeled_lab, pred)
    # print("Louvain:")
    # print(scores_louvain)

    # ======== 使用 KMeans 聚类评估聚类性能 ========
    # pred_kmcls = kmeans_cluster_infer(embeddings, unlabeled_idx, n_cls=len(set(info["cell_label"])))
    # scores_kmcls = compute_scores_for_cls(unlabeled_lab, pred_kmcls)
    # print("kmCLS:")
    # print(scores_kmcls)

    print("######################## over!")  # 所有流程结束

    # ======== 可视化（UMAP） ========
    # if args.visual:
    # if 1:
    #     em = anndata.AnnData(embeddings[unlabeled_idx])  # 创建 AnnData 对象
    #     # em.obs['cell_label'] = [info['cell_label'][i] for i in lab_full[unlabeled_idx]]  # 设置细胞标签
    #     sc.pp.neighbors(em, n_neighbors=30, use_rep='X')  # 构建图
    #     sc.tl.umap(em)  # 计算 UMAP
    #     sc.pl.umap(em, color=['cell_label'])  # 绘制图像
    #     plt.savefig("cls.png")
    #     plt.show()

    from utils import cluster_and_plot


    # embeddings = embeddings.values  # numpy array
    # 假设 adata 已经读入，embeddings 是 numpy array 对应 adata.obs_names
    adata_cluster=cluster_and_plot(adata, embeddings, method="louvain", resolution=1.0, title_suffix="scSemiAE")
    # adata=cluster_and_plot(adata, embeddings, method="kmeans", resolution=1.0, title_suffix="scSemiAE")

    
    # 初始化全 False
    user_sel = np.zeros(adata_cluster.n_obs, dtype=bool)
    # 把那些选中的索引置 True
    user_sel[cell_idx_list] = True
    # 写入 adata.obs
    adata_cluster.obs['user_selected'] = user_sel

    user_sel = np.zeros(adata.n_obs, dtype=bool)
    # 把那些选中的索引置 True
    user_sel[cell_idx_list] = True
    # 写入 adata.obs
    adata.obs['user_selected'] = user_sel


    from eval import plot_similar_n, compute_dist_metrics, plot_nearest_n


    print('adata user_selected  True数:', adata.obs['user_selected'].sum())
    print('adata_cluster True数:', adata_cluster.obs['user_selected'].sum())

    # 看看用户相关簇占比是否异常（如果接近所有簇，Nearest-N 会偏高）
    # for obj, name in [(adata, 'adata'), (adata_cluster, 'adata_cluster')]:
    #     if 'cluster' in obj.obs:
    #         labels = obj.obs['cluster'].astype(str)
    #         user_clusters = labels[obj.obs['user_selected']].unique()
    #         print(f'[{name}] 总簇数={labels.nunique()}，用户相关簇数={len(user_clusters)}')


    user_clusters_series = adata_cluster.obs.loc[adata_cluster.obs["user_selected"], "cluster"]

    print("用户选择的细胞及其聚类：")
    print(user_clusters_series)

    # 如果只想看有哪些簇（去重）
    user_cluster_list = user_clusters_series.unique()
    print("用户选择涉及的簇:", user_cluster_list)

    # 如果还想知道每个簇的用户细胞数量
    print("用户选择簇的计数：")
    print(user_clusters_series.value_counts())



    # adata.obs['annotation'] = adata.obs['annotation'].astype(str)
    plot_similar_n(adata, max_n=20, cluster_col='leiden-1')
    compute_dist_metrics(adata, cluster_col='leiden-1', plot=True)
    plot_nearest_n(adata, cluster_col='leiden-1')

    props = plot_similar_n(adata_cluster, max_n=20)
    metrics = compute_dist_metrics(adata_cluster, plot=True)
    n_list, results = plot_nearest_n(adata_cluster)








# 程序入口
if __name__ == "__main__":
    main()