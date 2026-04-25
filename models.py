import torch
import torch.nn as nn


# ==========================================
# 核心救场组件: 3D 深度可分离卷积
# 参数量仅为普通 3D 卷积的 1/10，从物理层面掐断过拟合
# ==========================================
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        # Depthwise: 每个通道独立卷积 (学习空间特征)
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1,
                                   stride=stride, groups=in_channels, bias=False)
        self.bn1 = nn.InstanceNorm3d(in_channels, affine=True)
        # Pointwise: 1x1x1 卷积 (学习跨通道特征)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        return x


class LightweightImageEncoder(nn.Module):
    def __init__(self, out_channels=128):  # 整体降维，拒绝冗余特征
        super(LightweightImageEncoder, self).__init__()
        # 第一层保留普通卷积，用于提取基础边缘
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 核心特征提取全部替换为深度可分离卷积
        self.layer1 = DepthwiseSeparableConv3d(16, 32, stride=2)
        self.layer2 = DepthwiseSeparableConv3d(32, 64, stride=2)
        self.layer3 = DepthwiseSeparableConv3d(64, 128, stride=2)
        self.layer4 = DepthwiseSeparableConv3d(128, out_channels, stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        # 空间 Dropout (Spatial Dropout)，直接丢弃整个通道的特征图
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=6, out_channels=64):
        super(ClinicalEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.mlp(x)


class CrossModalAttentionNetwork(nn.Module):
    def __init__(self, config=None):
        super(CrossModalAttentionNetwork, self).__init__()
        self.num_classes = config['model']['num_classes'] if config else 3
        self.clinical_dim = config['model']['clinical_dim'] if config else 6

        img_out_dim = 128
        clin_out_dim = 64
        hidden_dim = 128

        self.img_encoder = LightweightImageEncoder(out_channels=img_out_dim)
        self.clin_encoder = ClinicalEncoder(input_dim=self.clinical_dim, out_channels=clin_out_dim)

        self.img_proj = nn.Linear(img_out_dim, hidden_dim)
        self.clin_proj = nn.Linear(clin_out_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, img_x, clin_x):
        img_features = self.img_encoder(img_x)
        clin_features = self.clin_encoder(clin_x)

        i_h = torch.tanh(self.img_proj(img_features))
        c_h = torch.tanh(self.clin_proj(clin_features))

        joint_features = torch.cat([i_h, c_h], dim=1)
        z = self.gate(joint_features)

        fused_features = z * i_h + (1 - z) * c_h
        logits = self.classifier(fused_features)
        return logits