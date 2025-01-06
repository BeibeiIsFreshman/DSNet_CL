import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptiveMutualLearningLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        """
        初始化域适应相互学习损失
        Args:
            temperature: KL散度的温度参数，用于控制软标签的平滑程度
            alpha: 域适应loss的权重系数
        """
        super(DomainAdaptiveMutualLearningLoss, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def domain_adaptation_alignment(self, out1, out2):
        """
        域适应对齐模块：减少两个输出分布之间的差异
        使用最大平均差异(MMD)来度量分布差异
        """
        # 将特征展平为2D张量 [N, C*H*W]
        n = out1.size(0)
        out1_flat = out1.view(n, -1)
        out2_flat = out2.view(n, -1)

        # 计算核矩阵
        kernel_xx = self.gaussian_kernel_matrix(out1_flat, out1_flat)
        kernel_yy = self.gaussian_kernel_matrix(out2_flat, out2_flat)
        kernel_xy = self.gaussian_kernel_matrix(out1_flat, out2_flat)

        # 计算MMD距离
        mmd = kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()
        return mmd

    def gaussian_kernel_matrix(self, x, y , sigma=1.0):
        """
        计算高斯核矩阵
        """
        n_x = x.size(0)
        n_y = y.size(0)
        # print("x:", x)
        # print("y:", y)

        x_norm = x.pow(2).sum(1).view(-1, 1)
        y_norm = y.pow(2).sum(1).view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        # print("dist:", dist.shape)
        # print("dist:", dist)
        # print("torch.exp(-dist / (2.0 * sigma ** 2)):", torch.exp(-dist / (2.0 * sigma ** 2)))
        return torch.exp(-dist / (2.0 * sigma ** 2))

    def process_outputs(self, out1, out2):
        """
        处理网络输出，生成域适应后的输出
        """
        # 首先进行空间注意力处理
        attention1 = F.softmax(out1.sum(dim=1, keepdim=True), dim=2)
        attention2 = F.softmax(out2.sum(dim=1, keepdim=True), dim=2)

        # 应用注意力机制
        out1_ = out1 * attention2
        out2_ = out2 * attention1

        # 归一化处理
        out1_ = F.normalize(out1_, p=2, dim=1)
        out2_ = F.normalize(out2_, p=2, dim=1)

        return out1_, out2_

    def kl_divergence(self, out1_, out2_):
        """
        计算KL散度损失
        """
        log_prob1 = F.log_softmax(out1_ / self.temperature, dim=1)
        log_prob2 = F.log_softmax(out2_ / self.temperature, dim=1)
        prob1 = F.softmax(out1_ / self.temperature, dim=1)
        prob2 = F.softmax(out2_ / self.temperature, dim=1)

        # 双向KL散度
        kld_loss = (F.kl_div(log_prob1, prob2.detach(), reduction='batchmean') +
                    F.kl_div(log_prob2, prob1.detach(), reduction='batchmean')) / 2
        return kld_loss

    def forward(self, out1, out2):
        """
        前向传播，计算总损失
        Args:
            out1: 第一个网络的输出 [N,C,H,W]
            out2: 第二个网络的输出 [N,C,H,W]
        Returns:
            total_loss: 总损失值
        """
        # 1. 域适应对齐
        domain_loss = self.domain_adaptation_alignment(out1, out2)

        # 2. 处理输出
        out1_, out2_ = self.process_outputs(out1, out2)

        # 3. 计算KL散度损失
        kld_loss = self.kl_divergence(out1_, out2_)

        # 4. 合并损失
        total_loss = kld_loss + self.alpha * domain_loss

        return total_loss