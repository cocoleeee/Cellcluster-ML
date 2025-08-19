```python
# 数据与保存路径
parser.add_argument("-dpath", "--data_path", default="./dataset/", help="数据集文件夹路径")
parser.add_argument("-spath", "--save_path", default="./output/", help="输出目录路径")

# 有标签数据设置（数量或比例二选一）
parser.add_argument("-lsize", "--lab_size", type=int, default=10, help="每类细胞的有标签样本数（默认：10）")
parser.add_argument("-lratio", "--lab_ratio", type=float, default=-1, help="每类细胞的有标签样本比例（默认：-1，不启用）")

# 随机种子与训练相关参数
parser.add_argument("-s", "--seed", type=int, default=0, help="用于加载数据的随机种子（默认：0）")
parser.add_argument('--cuda', action='store_true', help='是否启用 CUDA（GPU）')
parser.add_argument('-pretrain_batch', '--pretrain_batch', type=int, default=100, help="预训练时的批大小（默认：100）")
parser.add_argument('-nepoch', '--epochs', type=int, default=60, help='训练轮数（默认：60）')
parser.add_argument('-nepoch_pretrain', '--epochs_pretrain', type=int, default=50, help='预训练轮数（默认：50）')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='学习率（默认：0.001）')
parser.add_argument('-lrS', '--lr_scheduler_step', type=int, default=10, help='学习率调度 step（默认：10）')
parser.add_argument('-lrG', '--lr_scheduler_gamma', type=float, default=0.5, help='学习率调度 gamma（默认：0.5）')
parser.add_argument('-lbd', '--Lambda', type=float, default=1, help='L2 正则化权重（默认：1）')
parser.add_argument('-v', '--visual', type=bool, default=False, help='是否可视化结果（默认：False）')
```

#### 数据处理（data.py)


##### 1. `__init__(...)`

**作用：**  
初始化 `Data` 类，读取表达矩阵和元数据文件，设定划分标签的方式（数量或比例）。

---

##### 2. `load_all(...)`

**作用：**  
加载所有数据，并根据参数决定哪部分标签“可见”（labeled）哪部分“隐藏”（unlabeled），模拟半监督情况。返回表达矩阵、完整标签、被保留的有标签索引、无标签索引，以及一些辅助信息。

**返回内容包括：**

- 读出表达矩阵为 `expr`（numpy 数组）。
- 用 `celllabel_to_numeric` 将指定的 label（如 `celltype`）映射成整数，并拿到完整标签 `lab_full`。
- 根据 `number_of_hide`：
    - 如果为 0，调用 `hide_labs`：按每类保留一定数目/比例标签。
    - 否则调用 `hide_some_labs`：只对部分类保留标签，其他类全部隐藏。
- 构造 `info` 字典包括
    - 原始的 label 映射（`cell_label`）、cell id、gene names。
    

---

##### 3. `hide_labs(...)`

对每个类别独立地，按照 `labeled_size`（或 `labeled_ratio` 计算后）取前若干个样本保留标签，其余归为无标签。


- 使用 `gen_idx_byclass` 把同类样本的索引聚在一起。
- 每类内部用固定的 `seed` 进行 shuffle，保证重复运行一致。
- 如果 `labeled_ratio` 存在，会覆盖 `self.labeled_size`（按当前类大小重新计算）。
- 如果 `labeled_size` >= 该类总数，则保留全部标签。
- 返回两个列表：`labeled_idx`（保留标签的样本索引）和 `unlabeled_idx`（隐藏标签的样本索引）。
    

---

##### 4. `hide_some_labs(...)`

模拟“只部分类别有标签”的情况：只对前 `number` 个（顺序打乱后）满足样本数 >= 50 的类保留标签，其余类全部隐藏；样本数 <50 的类全部隐藏。



- `number` 表示“有标签的类别个数”
- 使用 `random.Random(self.seed).shuffle(idx_byclass)` 试图打乱字典。
- 接下来对前 `number` 个符合条件的类做类似 `hide_labs` 的处理，对其他类直接把它们全部加入 `unlabeled_idx`。 
- 同样处理 `labeled_ratio` 覆盖 `self.labeled_size` 的逻辑。
    

---

![[Pasted image 20250804004624.png]]


**`ExperimentDataset`**

- 输入：表达矩阵、细胞 ID、标签（可为 None 或部分标签）
- 处理：稀疏矩阵转 dense，再转 tensor
- 提供：`__getitem__` 和 `__len__`，供 DataLoader 使用

