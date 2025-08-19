import torch
import pandas as pd
from model.loss import M1_loss
from model.losses import M2_loss
from model.net import FullNet
from model.utils import init_data_loaders
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from model.dataset import ExperimentDataset


import torch.nn.functional as F


import scanpy as sc
adata = sc.read_h5ad("sc_sampled.h5ad")

cell_idx_list = [7,10,25,37,47,61,72,81,93,103,107,119,120,126,127,144,171,200,213,220,251,254,257,258,260,271,275,276,282,293,305,308,309,312,346,376,378,379,425,427,437,438,452,460,469,494,497,501,516,535,538,540,559,563,578,584,595,601,605,614,632,637,650,652,656,661,662,665,685,707,711,730,733,738,742,743,772,773,775,776,778,784,804,810,812,839,840,851,854,860,868,869,884,896,905,908,912,920,921,941,949,965,967,970,971,978,981,1003,1013,1028,1047,1049,1052,1060,1077,1083,1087,1089,1092,1112,1113,1121,1125,1136,1148,1151,1162,1163,1165,1173,1181,1182,1192,1204,1208,1216,1217,1241,1258,1263,1268,1282,1294,1322,1336,1338,1341,1349,1350,1361,1364,1371,1372,1382,1384,1394,1407,1410,1417,1421,1436,1441,1464,1467,1473,1474,1475,1490,1509,1515,1517,1520,1527,1545,1546,1549,1567,1578,1581,1584,1587,1595,1603,1606,1607,1617,1619,1620,1621,1627,1630,1637,1642,1656,1672,1673,1678,1679,1680,1689,1710,1718,1728,1732,1733,1739,1751,1765,1769,1773,1781,1790,1801,1804,1816,1822,1828,1830,1835,1840,1850,1856,1858,1898,1911,1924,1937,1947,1948,1949,1961,1966,1970,1972,1973,1974,1975,1981,1983,1994,1997,2000,2001,2010,2014,2022,2023,2034,2046,2057,2064,2070,2075,2080,2085,2106,2133,2145,2150,2155,2174,2185,2189,2195,2196,2207,2213,2226,2229,2231,2249,2253,2255,2256,2271,2273,2277,2283,2285,2286,2290,2291,2295,2303,2312,2313,2317,2320,2323,2325,2327,2347,2351,2366,2376,2377,2384,2395,2397,2401,2403,2405,2406,2417,2421,2429,2441,2456,2464,2469,2472,2481,2483,2489,2497,2500,2503,2504,2521,2522,2525,2535,2539,2544,2548,2553,2556,2571,2575,2582,2583,2591,2615,2616,2617,2621,2623,2626,2629,2641,2648,2660,2661,2674,2676,2682,2688,2694,2696,2703,2705,2716,2721,2724,2727,2732,2734,2740,2743,2752,2761,2762,2765,2766,2776,2798,2802,2822,2828,2834,2838,2849,2870,2881,2888,2893,2911,2916,2917,2925,2926,2931,2936,2937,2942,2950,2951,2954,2961,2972,2976,2978,2981,2997,612,902,0,3,12,98,114,128,147,151,185,218,219,228,234,236,240,241,287,295,304,318,364,409,443,445,453,480,519,525,536,562,572,627,647,676,694,698,705,724,749,779,783,787,811,892,893,894,904,917,930,974,979,983,986,999,1004,1031,1038,1078,1081,1090,1100,1130,1168,1180,1185,1194,1197,1248,1256,1278,1326,1333,1340,1346,1347,1356,1365,1373,1389,1409,1429,1448,1449,1453,1476,1477,1479,1514,1565,1583,1616,1625,1632,1670,1676,1686,1702,1724,1741,1748,1749,1796,1823,1827,1839,1848,1854,1903,1906,1952,1969,1978,1986,1999,2005,2019,2021,2024,2084,2098,2099,2100,2190,2205,2216,2239,2247,2262,2294,2304,2322,2332,2336,2355,2413,2446,2448,2467,2478,2488,2491,2530,2536,2538,2545,2546,2560,2563,2665,2672,2698,2702,2777,2779,2788,2793,2801,2807,2812,2824,2825,2839,2840,2865,2878,2909,2924]
 
pos_idx = np.array(cell_idx_list, dtype=int)
pos_idx = pos_idx[(pos_idx >= 0) & (pos_idx < adata.n_obs)]
pos_names = adata.obs_names[pos_idx].tolist()     # ★ 与 Dataset 的 cells 类型一致（字符串）

    # 标记到 adata.obs 便于后续可视化/评估
user_sel = np.zeros(adata.n_obs, dtype=bool)
user_sel[pos_idx] = True
adata.obs['user_selected'] = user_sel

class scSemiAE:
    def __init__(self, params, labeled_data, pretrain_data, hid_dim_1=500, hid_dim_2=50, p_drop=0.1):
        """
        Parameters
		----------
        params: 
            parameters of the scSemiAE model
        labeled_data: 
            labeled dataset. 
        pretrain_data: 
            dataset for pretraining. 
        hid_dim_1: int
            dimension in the first layer of the network (default: 500)
        hid_dim_2: int
            dimension in the second layer of the network (default: 50)
        p_drop: float
            dropout probability (default: 0.1)
        """
        # prepare data
        train_load, pretrain_load = init_data_loaders(labeled_data, pretrain_data, params.pretrain_batch)
        self.train_loader = train_load
        self.pretrain_loader = pretrain_load

        # prepare network
        x_dim = self.pretrain_loader.dataset.get_dim()
        self.init_model(x_dim, hid_dim_1, hid_dim_2, p_drop, params.device)

        # prepare parameters
        self.device = params.device
        self.epochs = params.epochs
        self.epochs_pretrain = params.epochs_pretrain

        self.lr = params.learning_rate
        self.lr_gamma = params.lr_scheduler_gamma
        self.step_size = params.lr_scheduler_step

        self.Lambda = params.Lambda

        # === TensorBoard ===
        self.tb_writer = None
        if SummaryWriter is not None:
            logdir = getattr(params, "tb_logdir", "tb_log")
            if logdir is not None:
                try:
                    self.tb_writer = SummaryWriter(log_dir=logdir)
                except Exception:
                    self.tb_writer = None



    def init_model(self, x_dim, hid_dim_1, hid_dim_2, p_drop, device):
        """
        Initialize the model.

        Parameters
		----------
        x_dim: int 
            imput dimension
        hid_dim_1: int
            dimension in the first layer of the network
        hid_dim_2: int
            dimension in the second layer of the network
        p_drop: float
            dropout probability
        device: "cpu" or "cuda"
        """
        self.model = FullNet(x_dim, hid_dim_1, hid_dim_2, p_drop = p_drop).to(device)
        
    def pretrain(self, optim, plot_flag=False):
        """
        Pretraining the model.
        
        Parameters
		----------
        optim:
            an optimizer for pretraining stages
        plot_flag: bool
            Whether to show the loss function of pretraining stages
        """

        print("# pretraining:")
        loss_M1 = []
        for _ in range(self.epochs_pretrain):
            loss_batch = 0
            for idx, batch in enumerate(self.pretrain_loader):
                x, _, _ = batch
                x = x.to(self.device)
                _, decoded = self.model(x)
                loss = M1_loss(decoded, x)
                loss_batch += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
            idx += 1 # idx starts at 0 and need to add 1 
            loss_M1.append(loss_batch / idx)
        if plot_flag:
            i = np.arange(0, self.epochs_pretrain, 1)
            plt.plot(i, loss_M1, color='red')
            plt.savefig('M1.jpg')
            plt.show()


    def train(self, pretrain_flag=False, plot_flag=False):
        """
        Training the model.
        """
        # ====== 预训练 ======
        optim_pretrain = torch.optim.Adam(params=list(self.model.parameters()), lr=self.lr)
        self.pretrain(optim_pretrain, plot_flag)
        print("pretraining is over!")

        # ====== 参数准备 ======
        lr = self.lr
        Lambda = self.Lambda
        epochs = 0 if pretrain_flag else self.epochs

        # 只优化 encoder
        optim = torch.optim.Adam(params=list(self.model.encoder.parameters()), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim, gamma=self.lr_gamma, step_size=self.step_size
        )

        print("fine-tuning:")

        # 个性化损失的可选参数（若外部没设置，这里已有默认值）
        user_set       = getattr(self, 'user_set', None)               # set(str)
        alpha_user     = getattr(self, 'alpha_user', 3.0)
        beta_user      = getattr(self, 'beta_user', 3.0)
        use_triplet    = getattr(self, 'use_triplet', True)
        w_triplet      = getattr(self, 'w_triplet', 0.1)
        triplet_margin = getattr(self, 'triplet_margin', 0.5)
        use_supcon     = getattr(self, 'use_supcon', True)
        w_supcon       = getattr(self, 'w_supcon', 0.1)
        supcon_tau     = getattr(self, 'supcon_tau', 0.1)
        softmin_tau    = getattr(self, 'softmin_tau', 0.5)

        # 为了画旧版三条曲线（M2/L1/L2），仍保留这三个容器
        loss_M2, loss_L1, loss_L2 = [], [], []

        # 我们还会统计每个 epoch 的所有分量，便于打印/TensorBoard
        keys = [
            'L_total','L1','L2','L_pull','L_push',
           
        ]

        refresh_every = getattr(self, "refresh_every", 10)   # 你也可以放到 args 里
        K_pseudo      = getattr(self, "K_pseudo", 10)
        self._refresh_pseudo_labels(K=K_pseudo)



        for epoch in range(epochs):
            stats_sum = {k: 0.0 for k in keys}
            n_batches = 0

            for idx, batch in enumerate(self.train_loader):
                x, y, cells = batch  # 确保第三个返回的是字符串 cell id

                # 调试：仅首个 epoch 的前几个 batch 打印用户命中
                if epoch == 0 and idx < 3 and user_set is not None:
                    hits = len(set(cells) & set(user_set))
                    print(f"[debug] epoch{epoch} batch{idx}: user hits = {hits}/{len(cells)}")

                x = x.to(self.device)
                y = y.to(self.device)
                if y.dtype != torch.long:
                    y = y.long()

                encoded, _ = self.model(x)

                # ====== curriculum: 逐步增权（按 epoch 分段）======
                ep = epoch + 1
                if getattr(self, 'curriculum', True):
                    if   ep <= 3:  current_w_pull, current_w_push = 0.2, 0.0
                    elif ep <= 7:  current_w_pull, current_w_push = 0.4, 0.2
                    else:          current_w_pull, current_w_push = 0.6, 0.3
                else:
                    current_w_pull, current_w_push = 0.4, 0.2  # 固定值兜底

                # ====== 自适应 margin：基于当前 batch 的非用户→用户质心距离的分位数 ======
                # if getattr(self, 'use_adaptive_margin', True) and (user_set is not None):
                #     with torch.no_grad():
                #         # batch 内哪些是用户样本
                #         umask = torch.tensor([(c in user_set) for c in cells], device=self.device)
                #         if umask.any() and (~umask).any():
                #             # 只对“用户拉/推”分支用单位化的表示（更像余弦）
                #             z_u = F.normalize(encoded[umask], dim=1)
                #             z_b = F.normalize(encoded[~umask], dim=1)
                #             c_u = z_u.mean(0, keepdim=True)  # (1, D)
                #             d   = torch.cdist(z_b, c_u, p=2).squeeze(1)  # 非用户→用户质心距离
                #             # 取 30% 分位做 margin，夹到 [0.3, 0.9]
                #             m = torch.quantile(d, q=0.30).clamp_(0.3, 0.9).item()
                #         else:
                #             m = 0.6  # 若 batch 没有足够的用户/非用户，给个兜底
                # else:
                #     m = 0.6

                # （可选）前几个 batch 打印看看当前权重与 margin
                # if epoch == 0 and idx < 2:
                #     print(f"[debug] ep{ep} batch{idx} w_pull={current_w_pull:.2f} "
                #         f"w_push={current_w_push:.2f} margin={m:.2f}")

                # === 新版 M2_loss：返回 (total_loss, comps_dict) ===
                total_loss, comps = M2_loss(
                    encoded, y, Lambda,
                    cells=cells,
                    user_set=user_set
                    # ,w_pull=current_w_pull, w_push=current_w_push, margin=m
                )
                
                # 反传
                optim.zero_grad()
                total_loss.backward()
                optim.step()

                # 累计统计值
                for k in keys:
                    stats_sum[k] += comps[k]
                n_batches += 1

            # === 每个 epoch 结束：做均值 ===
            means = {k: (stats_sum[k] / max(n_batches, 1)) for k in keys}

            # 兼容你原来的三条曲线（M2/L1/L2）：用合成项
            # loss_M2.append(means['L_total'])
            # loss_L1.append(means['L1_total'])
            # loss_L2.append(means['L2_total'])

            # 学习率调度每个 epoch 再 step（更常见）
            lr_scheduler.step()

            # 打印
            print(f"[Epoch {epoch+1:03d}/{epochs}] "
                f"total={means['L_total']:.4f} "
                f"L1={means['L1']:.4f} L2={means['L2']:.4f} "
                f"L1u={means['L_pull']:.4f} L2u={means['L_push']:.4f} "
                f"lr={optim.param_groups[0]['lr']:.2e}")

            # 可选：TensorBoard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('train/L_total',   means['L_total'],   epoch)
                self.tb_writer.add_scalar('train/L1',        means['L1'],        epoch)
                self.tb_writer.add_scalar('train/L2',        means['L2'],        epoch)
                self.tb_writer.add_scalar('train/L_pull',   means['L_pull'],   epoch)
                self.tb_writer.add_scalar('train/L_push',   means['L_push'],   epoch)
                self.tb_writer.add_scalar('train/lr',        optim.param_groups[0]['lr'], epoch)

            if (epoch + 1) % refresh_every == 0:
                self._refresh_pseudo_labels(K=K_pseudo)
            

        # 训练结束：收尾
        if epochs > 0 and plot_flag:
            i = np.arange(0, epochs, 1)
            plt.plot(i, loss_M2, color='red',   label="M2 (L_total)")
            plt.plot(i, loss_L1, color='blue',  label='L1_total')
            plt.plot(i, loss_L2, color='green', label='L2_total')
            plt.legend(loc=0, ncol=1)
            plt.savefig('M2.jpg')
            plt.show()

        if self.tb_writer is not None:
            self.tb_writer.flush()

        embeddings = self.compute_embeddings()
        return embeddings



    def compute_embeddings(self):
        """
        Saving embeddings from the dataset
        Returns
		----------
        embeddings: dataframe
            low dimensional representations for cells
        """

        embeddings_all = []

        for _, batch in enumerate(self.pretrain_loader):
            x, y, cells = batch
            x = x.to(self.device)
            encoded, _ = self.model(x)
            embeddings_all.append(pd.DataFrame(encoded.cpu().detach().numpy(), index = cells))
        
        # correspondding with labels
        embeddings = pd.concat(embeddings_all, axis=0).sort_index()

        return embeddings
    
    @torch.no_grad()
    def _refresh_pseudo_labels(self, K=5):
        """
        用当前 encoder 在全体样本上生成伪标签，并把用户细胞并到同一“主簇”，
        然后重建 self.train_loader 让后续 L1/L2 在全体样本上起作用。
        """
        self.model.eval()

        # 2.1 聚齐全体样本（保持顺序），以及其细胞ID
        xs, cells_all = [], []
        for x, _, cells in self.pretrain_loader:
            xs.append(x)                    # 这里是原表达，不是embedding
            cells_all.extend(list(cells))
        X_full = torch.cat(xs, dim=0).to(self.device)   # (N, G)

        # 2.2 提 embedding（当前 encoder）
        Z_full = self.model.encoder(X_full).detach().cpu().numpy()  # (N, D)

        # 2.3 KMeans 产生伪标签
        km = KMeans(n_clusters=K, n_init=10, random_state=0)
        y_pseudo = km.fit_predict(Z_full)  # ndarray, shape=(N,)

        # 2.4 “用户并簇”：把用户细胞的簇统一到用户细胞中最常见的那个簇
        user_set = getattr(self, "user_set", None)
        if user_set:
            import numpy as np
            user_mask = np.array([c in user_set for c in cells_all], dtype=bool)
            if user_mask.any():
                # 找到用户细胞里占比最多的簇作为主簇
                labels_u = y_pseudo[user_mask]
                # 可能用户很少、甚至只有1个，也没关系
                # 若 labels_u 为空不会进入这里
                vals, cnts = np.unique(labels_u, return_counts=True)
                main_label = vals[cnts.argmax()]
                y_pseudo[user_mask] = main_label

        # 2.5 用“全体样本 + 伪标签(含用户并簇)”重建新的训练集/loader
        #    —— 注意：这里训练输入仍是“原表达 X_full”，不是 Z_full
        lab_tensor = X_full.detach().cpu().float()
        lab_names  = cells_all
        lab_labels = y_pseudo.astype(int).tolist()
        new_dataset = ExperimentDataset(lab_tensor, lab_names, lab_labels)
        self.train_loader = DataLoader(new_dataset, shuffle=True, batch_size=100)

        self.model.train()
