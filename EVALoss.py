import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Train_Text_Val.Config_loss import pytorch_iou
iou_loss = pytorch_iou.IOU(size_average=True)



class AdvancedSPLFeatureLoss(nn.Module):
    def __init__(
            self,
            channels_list=[64, 128, 320, 512],
            init_threshold=0.5,
            growth_rate=1.05,  # 降低增长率避免阈值增长过快
            min_weight=0.1,
            eps=1e-6  # 添加数值稳定性参数
    ):
        """
        改进的SPL特征损失计算模块

        参数:
            channels_list: 特征通道数列表
            init_threshold: SPL初始阈值
            growth_rate: 阈值增长率 (降低以避免过快增长)
            min_weight: 特征权重的最小值
            eps: 数值稳定性常数
        """
        super(AdvancedSPLFeatureLoss, self).__init__()

        self.threshold = init_threshold
        self.growth_rate = growth_rate
        self.min_weight = min_weight
        self.eps = eps

        # 使用Parameter注册特征权重
        self.feature_weights = nn.Parameter(torch.ones(len(channels_list)))
        self.softmax = nn.Softmax(dim=0)

        # 改进的特征难度评估器
        self.difficulty_estimator = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, ch // 16, 1),
                nn.BatchNorm2d(ch // 16),  # 添加BN层提高稳定性
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 16, 1, 1),
                nn.Sigmoid()
            ) for ch in channels_list
        ])

    def normalize_feature(self, x):
        """特征标准化"""
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + self.eps
        x = (x - mean) / std
        return x.view(b, c, h, w)

    def estimate_difficulty(self, feature):
        """改进的特征难度评估方法"""
        # 首先标准化特征
        # feature = self.normalize_feature(feature)

        # 计算多个统计指标并确保数值稳定性
        spatial_variance = torch.var(feature, dim=[2, 3], keepdim=True).clamp(min=self.eps)
        channel_variance = torch.var(feature, dim=[1], keepdim=True).clamp(min=self.eps)

        # 使用L1范数代替偏度，避免数值不稳定
        spatial_l1 = torch.mean(torch.abs(feature), dim=[2, 3], keepdim=True)
        channel_l1 = torch.mean(torch.abs(feature), dim=[1], keepdim=True)

        # 组合难度指标并应用log缩放
        difficulty = torch.log(1 + spatial_variance) * torch.log(1 + channel_variance) * \
                     spatial_l1 * channel_l1

        return difficulty.mean().clamp(max=100)  # 限制最大难度值

    def get_spl_weight(self, difficulty):
        """改进的SPL权重计算"""
        # 使用softer的权重计算方式
        normalized_diff = difficulty / (self.threshold + self.eps)
        weight = torch.exp(-normalized_diff)
        return weight.clamp(min=self.min_weight)

    def compute_feature_similarity(self, f1, f2):
        """计算特征相似度"""
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)
        return F.cosine_similarity(f1, f2, dim=1).mean()

    def forward(self, dnet_features, snet_features, current_epoch):
        """改进的前向传播"""
        total_loss = 0
        batch_weights = []

        # 归一化特征层权重
        norm_weights = self.softmax(self.feature_weights).clamp(min=self.min_weight)

        for idx, (dnet_feat, snet_feat) in enumerate(zip(dnet_features, snet_features)):
            # 特征预处理
            dnet_feat = self.normalize_feature(dnet_feat)
            snet_feat = self.normalize_feature(snet_feat)

            # 评估难度
            d_difficulty = self.estimate_difficulty(dnet_feat)
            s_difficulty = self.estimate_difficulty(snet_feat)

            # 计算SPL权重
            d_weights = self.get_spl_weight(d_difficulty)
            s_weights = self.get_spl_weight(s_difficulty)

            # 计算特征重要性
            d_importance = self.difficulty_estimator[idx](dnet_feat)
            s_importance = self.difficulty_estimator[idx](snet_feat)

            # 结合KL散度和余弦相似度的损失
            # kl_loss = 0.5 * (
            #         F.kl_div(
            #             F.log_softmax(dnet_feat.flatten(2), dim=-1),
            #             F.softmax(snet_feat.detach().flatten(2), dim=-1),
            #             reduction='batchmean'
            #         ) +
            #         F.kl_div(
            #             F.log_softmax(snet_feat.flatten(2), dim=-1),
            #             F.softmax(dnet_feat.detach().flatten(2), dim=-1),
            #             reduction='batchmean'
            #         )
            # )
            kl_loss = iou_loss(dnet_feat, snet_feat.detach()) + iou_loss(snet_feat, dnet_feat.detach())

            # 添加余弦相似度损失
            cos_loss = 1 - self.compute_feature_similarity(dnet_feat, snet_feat)

            # 组合损失
            layer_loss = (kl_loss + cos_loss) * \
                         (d_weights * s_weights).mean() * \
                         (d_importance * s_importance).mean()

            total_loss += norm_weights[idx] * layer_loss

            batch_weights.append((d_weights * s_weights).mean().item())

        # 动态更新难度阈值，使用指数移动平均
        if current_epoch % 5 == 0:
            self.threshold = min(
                self.threshold * self.growth_rate,
                max(batch_weights) * 2  # 确保阈值不会增长过快
            )

        return total_loss