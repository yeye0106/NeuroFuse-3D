import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 核心特征提取：轻量级深度可分离卷积 + SE 通道注意力
class LightweightSEBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 深度可分离卷积 (防过拟合)
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1,
                                   stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # SE 通道注意力
        self.se_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels, bias=False),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)

        # 应用 SE 注意力
        b, c, _, _, _ = x.size()
        se_weight = self.se_avg_pool(x).view(b, c)
        se_weight = self.se_fc(se_weight).view(b, c, 1, 1, 1)
        x = x * se_weight.expand_as(x)

        return x + residual


class ImageEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer1 = LightweightSEBlock3D(16, 32, stride=2)
        self.layer2 = LightweightSEBlock3D(32, 64, stride=2)
        self.layer3 = LightweightSEBlock3D(64, 128, stride=2)
        self.layer4 = LightweightSEBlock3D(128, out_channels, stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
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
        super().__init__()
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


# 2. 跨模态融合网络 (GMU 门控机制)
class CrossModalAttentionNetwork(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.num_classes = config['model']['num_classes'] if config else 3
        self.clinical_dim = config['model']['clinical_dim'] if config else 6

        img_dim = 128
        clin_dim = 64
        hidden_dim = 128

        self.img_encoder = ImageEncoder(out_channels=img_dim)
        self.clin_encoder = ClinicalEncoder(input_dim=self.clinical_dim, out_channels=clin_dim)

        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.clin_proj = nn.Linear(clin_dim, hidden_dim)

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
        return self.classifier(fused_features)