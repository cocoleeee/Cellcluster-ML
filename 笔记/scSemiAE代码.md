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
加载表达矩阵 + 标签信息，并按设置划分出有标签样本和无标签样本的索引，同时返回包含细胞ID、标签映射等的额外信息。

**返回内容包括：**

- `expr`：表达矩阵（NumPy数组）
    
- `lab_full`：所有细胞的数字标签
    
- `labeled_idx`：有标签样本索引
    
- `unlabeled_idx`：无标签样本索引
    
- `info`：元信息（标签映射、细胞ID、基因名等）
    

---

##### 3. `hide_labs(...)`

**作用：**  
根据每类样本的比例或数量，隐藏部分标签，模拟半监督学习。适合**“所有类别都有标签但不完整”**的情况。

**输入：**

- `lab`：所有样本的标签（整数形式）
    

**输出：**

- 有/无标签样本的索引
    

---

##### 4. `hide_some_labs(...)`

**作用：**  
进一步模拟更困难的情形：只对**部分细胞类型**保留标签，其他类型完全无标签，常用于**零样本或小样本学习**。

**输入：**

- `lab`：所有样本的标签   
- `number`：想要保留标签的细胞类别数量    

**输出：**

- 有/无标签样本的索引
    

---

## 🛠️ 辅助函数（外部）

### 5. `gen_idx_byclass(labels)`

**作用：**  
将标签列表分类，返回一个字典：  
`{class_label: [该类对应的样本索引]}`

便于后续按类别进行标签隐藏或划分。

---

### 6. `celllabel_to_numeric(celllabel)`

**作用：**  
将原始的细胞类型（字符串）转换为整数形式（模型可接受），同时生成数字与原始标签之间的映射关系。

**返回：**

- `mapping`：整数 ↔ 细胞标签 映射字典
    
- `truth_labels`：转为数字的标签数组（NumPy）
    

---

### 7. `techtype_to_numeric(batch_name)`

**作用：**  
将批次信息（如测序平台名等字符串）转换为整数标签，用于后续 batch effect 处理或可视化。

**返回：**

- `mapping`：整数 ↔ 批次名 映射字典
    
- `truth_batches`：转为整数的批次数组（NumPy）
    

---

