"""
Network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # <- for AAM head normalization

def full_block(in_features, out_features, p_drop):
    """
    full connected layer
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )

# -------- AAM head: 2-class Additive Angular Margin Softmax (background vs. user) --------
class AAM2Head(nn.Module):
    """
    二分类（y∈{0,1}）的角度分类头：
      - 对正类加入角度边际 margin，放大类间间隔；
      - logits = scale * (cos(theta_y) - margin_for_y)；
      - 训练时配合 CrossEntropy 使用即可。
    """
    def __init__(self, feat_dim: int, scale: float = 24.0, margin: float = 0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(2, feat_dim))  # 2 类（背景/用户）
        nn.init.xavier_normal_(self.W)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D)  未归一化的 encoder 输出
        y: (B,)    0/1 标签（long 或 bool 可转换）
        return: (B, 2) logits
        """
        z_n = F.normalize(z, dim=1)                 # 单位球
        W_n = F.normalize(self.W, dim=1)            # 权重也单位球
        cos = z_n @ W_n.t()                         # (B,2)
        # 对真实类别减去 margin（等价于在角度上加边际）
        y = y.long()
        m = torch.zeros_like(cos)
        m[torch.arange(z.size(0), device=z.device), y] = self.margin
        logits = self.scale * (cos - m)
        return logits
# -----------------------------------------------------------------------------------------

class FullNet(nn.Module):
    """
    the backbone of the autoencoder of pretraining stages
    """
    def __init__(self, x_dim, hid_dim=64, z_dim=64, p_drop=0.2):
        super(FullNet, self).__init__()
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            full_block(x_dim, hid_dim, p_drop),
            full_block(hid_dim, z_dim, p_drop),
        )
        
        self.decoder = nn.Sequential(
            full_block(z_dim, hid_dim, p_drop),
            full_block(hid_dim, x_dim, p_drop),
        )

        # ---- AAM head (不会改变 forward 返回；训练阶段单独调用) ----
        self.aam_head = AAM2Head(feat_dim=z_dim, scale=24.0, margin=0.30)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
