# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""基于 CBAM 的多模态特征融合模块。"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("ChannelAttention", "SpatialAttention", "CBAMFusion")


class ChannelAttention(nn.Module):
    """通道注意力模块。

    使用平均池化和最大池化对通道特征进行重新校准。

    属性:
        avg_pool (nn.AdaptiveAvgPool2d): 全局平均池化层。
        max_pool (nn.AdaptiveMaxPool2d): 全局最大池化层。
        fc1 (nn.Conv2d): 第一个 1x1 卷积，用于通道降维。
        relu (nn.ReLU): ReLU 激活函数。
        fc2 (nn.Conv2d): 第二个 1x1 卷积，用于通道升维。
        sigmoid (nn.Sigmoid): Sigmoid 激活函数，生成注意力权重。

    示例:
        >>> ca = ChannelAttention(ch=64)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = ca(x)
        >>> print(out.shape)
        torch.Size([1, 64, 32, 32])
    """

    def __init__(self, ch: int, reduction: int = 16):
        """初始化通道注意力模块。

        参数:
            ch (int): 输入通道数。
            reduction (int): 通道降维比例，用于瓶颈结构。
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP，采用瓶颈结构
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用通道注意力。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        返回:
            (torch.Tensor): 应用通道注意力后的输出张量，形状为 (B, C, H, W)。
        """
        # 平均池化分支
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # 最大池化分支
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # 合并并应用 sigmoid
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力模块。

    沿通道维度使用平均池化和最大池化来应用空间注意力。

    属性:
        conv (nn.Conv2d): 7x7 卷积，用于空间注意力。
        sigmoid (nn.Sigmoid): Sigmoid 激活函数，生成注意力权重。

    示例:
        >>> sa = SpatialAttention()
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = sa(x)
        >>> print(out.shape)
        torch.Size([1, 64, 32, 32])
    """

    def __init__(self, kernel_size: int = 7):
        """初始化空间注意力模块。

        参数:
            kernel_size (int): 卷积核大小（默认: 7）。
        """
        super().__init__()
        assert kernel_size in {3, 7}, "卷积核大小必须是 3 或 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用空间注意力。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        返回:
            (torch.Tensor): 应用空间注意力后的输出张量，形状为 (B, C, H, W)。
        """
        # 沿通道维度进行平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维度进行最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接并应用卷积
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAMFusion(nn.Module):
    """基于 CBAM 的多模态融合模块。

    使用卷积块注意力模块 (CBAM) 融合 RGB 和 IR 特征，以增强多模态特征表示。

    该模块通过以下步骤处理 4 通道输入（3 RGB + 1 IR）：
    1. 使用 1x1 卷积分别从 RGB 和 IR 中提取特征
    2. 拼接特征
    3. 应用通道注意力来重新校准特征通道
    4. 应用空间注意力来强调重要的空间区域
    5. 将维度降回目标通道数

    属性:
        conv_rgb (nn.Conv2d): 用于 RGB 特征提取的 1x1 卷积。
        conv_ir (nn.Conv2d): 用于 IR 特征提取的 1x1 卷积。
        bn1 (nn.BatchNorm2d): 特征提取后的批归一化。
        channel_attention (ChannelAttention): 通道注意力模块。
        spatial_attention (SpatialAttention): 空间注意力模块。
        conv_fusion (nn.Conv2d): 用于降维的 1x1 卷积。
        bn2 (nn.BatchNorm2d): 融合后的批归一化。
        act (nn.SiLU): SiLU 激活函数。

    示例:
        >>> fusion = CBAMFusion(ch=64)
        >>> x = torch.randn(2, 4, 640, 640)  # 4通道输入 (RGB+IR)
        >>> out = fusion(x)
        >>> print(out.shape)
        torch.Size([2, 64, 640, 640])
    """

    def __init__(self, ch: int = 64, reduction: int = 16):
        """初始化 CBAM 融合模块。

        参数:
            ch (int): 输出通道数。
            reduction (int): 通道注意力瓶颈的通道降维比例。
        """
        super().__init__()
        # 特征提取层
        self.conv_rgb = nn.Conv2d(3, ch, 1, bias=False)  # RGB: 3 -> ch
        self.conv_ir = nn.Conv2d(1, ch, 1, bias=False)   # IR: 1 -> ch
        self.bn1 = nn.BatchNorm2d(ch * 2)

        # CBAM 注意力模块
        self.channel_attention = ChannelAttention(ch * 2, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # 融合层
        self.conv_fusion = nn.Conv2d(ch * 2, ch, 1, bias=False)  # 2*ch -> ch
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 4 通道输入应用基于 CBAM 的融合。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, 4, H, W)，其中前 3 个通道是 RGB，
                第 4 个通道是 IR。

        返回:
            (torch.Tensor): 融合后的特征张量，形状为 (B, ch, H, W)。

        异常:
            AssertionError: 如果输入不是 4 通道。
        """
        assert x.shape[1] == 4, f"期望 4 通道输入 (RGB+IR)，但得到 {x.shape[1]} 个通道"

        # 步骤 1: 分离 RGB 和 IR 通道
        x_rgb = x[:, :3, :, :]  # [B, 3, H, W]
        x_ir = x[:, 3:4, :, :]  # [B, 1, H, W]

        # 步骤 2: 分别提取特征
        feat_rgb = self.conv_rgb(x_rgb)  # [B, ch, H, W]
        feat_ir = self.conv_ir(x_ir)     # [B, ch, H, W]

        # 步骤 3: 拼接特征
        feat_concat = torch.cat([feat_rgb, feat_ir], dim=1)  # [B, 2*ch, H, W]
        feat_concat = self.bn1(feat_concat)

        # 步骤 4: 应用 CBAM 注意力
        feat_ca = self.channel_attention(feat_concat)  # 通道注意力
        feat_sa = self.spatial_attention(feat_ca)      # 空间注意力

        # 步骤 5: 降维和激活
        feat_fused = self.conv_fusion(feat_sa)  # [B, ch, H, W]
        feat_fused = self.bn2(feat_fused)
        out = self.act(feat_fused)

        return out


if __name__ == "__main__":
    """测试 CBAM 融合模块。"""
    print("测试 CBAM 融合模块...")

    # 测试通道注意力
    print("\n1. 测试通道注意力:")
    ca = ChannelAttention(ch=64)
    x_ca = torch.randn(2, 64, 32, 32)
    out_ca = ca(x_ca)
    print(f"   输入形状: {x_ca.shape}")
    print(f"   输出形状: {out_ca.shape}")
    assert out_ca.shape == x_ca.shape, "通道注意力输出形状不匹配！"
    print("   ✅ 通道注意力测试通过！")

    # 测试空间注意力
    print("\n2. 测试空间注意力:")
    sa = SpatialAttention()
    x_sa = torch.randn(2, 64, 32, 32)
    out_sa = sa(x_sa)
    print(f"   输入形状: {x_sa.shape}")
    print(f"   输出形状: {out_sa.shape}")
    assert out_sa.shape == x_sa.shape, "空间注意力输出形状不匹配！"
    print("   ✅ 空间注意力测试通过！")

    # 测试 CBAM 融合
    print("\n3. 测试 CBAM 融合:")
    fusion = CBAMFusion(ch=64)
    x_fusion = torch.randn(2, 4, 640, 640)  # 4通道输入 (RGB+IR)
    out_fusion = fusion(x_fusion)
    print(f"   输入形状: {x_fusion.shape}")
    print(f"   输出形状: {out_fusion.shape}")
    assert out_fusion.shape == (2, 64, 640, 640), "CBAM融合输出形状不匹配！"
    print("   ✅ CBAM融合测试通过！")

    # 测试不同通道数
    print("\n4. 测试不同通道数的 CBAM 融合:")
    for ch in [32, 64, 128]:
        fusion_test = CBAMFusion(ch=ch)
        x_test = torch.randn(1, 4, 320, 320)
        out_test = fusion_test(x_test)
        print(f"   ch={ch}: {x_test.shape} -> {out_test.shape}")
        assert out_test.shape == (1, ch, 320, 320), f"ch={ch} 时测试失败"
    print("   ✅ 所有通道数测试通过！")

    print("\n🎉 所有测试成功通过！")

