import torch
import torch.nn as nn
import torch.nn.functional as F


class UAMFusion(nn.Module):
    """
    UAM (Uncertainty-Aware Module) 融合模块
    设计思路：
    1. 输入 4 通道，拆分为 RGB(3) 和 IR(1)。
    2. 分别进行卷积提取初步特征。
    3. 使用通道注意力 (Global Pooling + MLP) 计算 '不确定性权重'。
    4. 权重互补：Weight_RGB + Weight_IR = 1 (类似 Softmax)。
    5. 如果 RGB 是全黑（不确定性高），网络会自动学习降低 RGB 权重，提高 IR 权重。
    """

    def __init__(self, c1, c2, k=3, s=2, g=1, d=1):
        super().__init__()
        self.conv_rgb = nn.Conv2d(3, c2, k, s, k // 2, bias=False)
        self.conv_ir = nn.Conv2d(1, c2, k, s, k // 2, bias=False)

        self.bn = nn.BatchNorm2d(c2)  # 融合后的 BN
        self.act = nn.SiLU()


        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化 [B, 2*c2, 1, 1]
            nn.Conv2d(c2 * 2, c2 // 2, 1),  # 降维/压缩
            nn.ReLU(),
            nn.Conv2d(c2 // 2, 2, 1),  # 升维到 2 (分别对应 RGB权重 和 IR权重)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [Batch, 4, H, W]

        # --- A. 拆分与预处理 ---
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        # --- B. 提取特征 ---
        feat_rgb = self.conv_rgb(x_rgb)  # [B, c2, H/2, W/2]
        feat_ir = self.conv_ir(x_ir)  # [B, c2, H/2, W/2]

        # --- C. 计算不确定性权重 ---
        # 拼接特征用于判断
        cat_feat = torch.cat([feat_rgb, feat_ir], dim=1)  # [B, 2*c2, ...]

        # 计算权重 [B, 2, 1, 1]
        # w[:, 0] 是 RGB 的权重, w[:, 1] 是 IR 的权重
        weights = self.attention(cat_feat)
        w_rgb = weights[:, 0:1, :, :]
        w_ir = weights[:, 1:2, :, :]

        # --- D. 加权融合 ---
        # 广播机制：权重会自动应用到所有通道
        feat_fused = (feat_rgb * w_rgb) + (feat_ir * w_ir)

        # --- E. 归一化与激活 ---
        return self.act(self.bn(feat_fused))