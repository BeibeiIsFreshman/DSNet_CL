import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from thop import profile
from Mine_Net.PVTv2 import pvt_v2_b4


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class RGB_Decoder(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], out_channels=64):
        super(RGB_Decoder, self).__init__()
        # 多尺度特征提取
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for i in range(4)
        ])

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 伪真值图生成
        self.pseudo_gt_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 64, kernel_size=1),
            nn.Sigmoid()
        )

        # 伪边缘图生成
        self.pseudo_edge_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 64, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # 多尺度特征提取
        multi_scale_features = [
            F.interpolate(self.multi_scale_convs[i](features[i]),
                          size=features[0].shape[2:],
                          mode='bilinear',
                          align_corners=False)
            for i in range(4)
        ]

        # 特征融合
        fused_features = torch.cat(multi_scale_features, dim=1)
        attention_weights = self.attention(fused_features)
        attended_features = sum([attention_weights[:, i:i + 1] * multi_scale_features[i] for i in range(4)])

        # 生成伪真值图
        pseudo_gt = self.pseudo_gt_conv(attended_features)

        # 生成初步伪边缘图
        pseudo_edge_initial = self.pseudo_edge_conv(attended_features)

        return pseudo_gt, pseudo_edge_initial

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class FusionModule(nn.Module):
    def __init__(self, channels):
        super(FusionModule, self).__init__()

        rgb_channels = channels
        depth_channels = channels
        out_channels = channels

        self.rgb_conv = nn.Conv2d(rgb_channels, out_channels, kernel_size=3, padding=1)
        self.depth_conv = nn.Conv2d(depth_channels, out_channels, kernel_size=3, padding=1)

        self.channel_attention = ChannelAttention(out_channels * 2)
        self.spatial_attention = SpatialAttention()

        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        # 多尺度特征提取
        self.multiscale_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=4, dilation=4),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=8, dilation=8)
        ])

        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_conv(rgb)
        depth_feat = self.depth_conv(depth)

        # 特征拼接
        concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)

        # 通道注意力
        channel_weight = self.channel_attention(concat_feat)
        concat_feat = concat_feat * channel_weight

        # 空间注意力
        spatial_weight = self.spatial_attention(concat_feat)
        concat_feat = concat_feat * spatial_weight

        # 特征融合
        fused_feat = self.fusion_conv(concat_feat)

        # 多尺度特征提取
        multiscale_feats = [conv(fused_feat) for conv in self.multiscale_convs]
        multiscale_feat = torch.cat(multiscale_feats, dim=1)

        # 最终输出
        output = self.output_conv(multiscale_feat)

        return output


class SearchModule(nn.Module):
    def __init__(self, fm_channels):
        super(SearchModule, self).__init__()
        self.fm_channels = fm_channels

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(fm_channels + 128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Conv2d(fm_channels + 128, fm_channels, kernel_size=1)

        # 空间搜索
        self.spatial_search = nn.Conv2d(fm_channels, fm_channels, kernel_size=3, padding=1, groups=fm_channels)

        # 通道搜索
        self.channel_search = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fm_channels, fm_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(fm_channels // 4, fm_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, fm, ff, fe):
        # 调整 ff 和 fe 的大小以匹配 fm
        ff_resized = F.interpolate(ff, size=fm.shape[2:], mode='bilinear', align_corners=False)
        fe_resized = F.interpolate(fe, size=fm.shape[2:], mode='bilinear', align_corners=False)

        # 拼接特征
        concat_features = torch.cat([fm, ff_resized, fe_resized], dim=1)

        # 注意力机制
        attention_map = self.attention(concat_features)
        attended_features = concat_features * attention_map

        # 特征融合
        fused_features = self.fusion(attended_features)

        # 空间搜索
        spatial_searched = self.spatial_search(fused_features)

        # 通道搜索
        channel_weights = self.channel_search(spatial_searched)
        channel_searched = spatial_searched * channel_weights

        return channel_searched


class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in1, in2, in3, in4):
        super(Decoder, self).__init__()
        self.bcon4 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4, in4, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3 = BasicConv2d(in3, in4, kernel_size=3, stride=1, padding=1)
        self.bcon2 = BasicConv2d(in2, in3, kernel_size=3, stride=1, padding=1)
        self.bcon1 = BasicConv2d(in_planes=in1, out_planes=in2, kernel_size=1, stride=1, padding=0)

        self.bcon4_3 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4 * 2, in3, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3_2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3 * 2, in2, kernel_size=3, stride=1, padding=1)
        )
        self.bcon2_1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )

        self.conv_d1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3, in2, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d3 = BasicConv2d(in2, in1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        f[3] = self.bcon4(f[3])
        f[2] = self.bcon3(f[2])
        f[1] = self.bcon2(f[1])
        f[0] = self.bcon1(f[0])

        d43 = self.bcon4_3(torch.cat((f[3], f[2]), 1))
        d32 = self.bcon3_2(torch.cat((d43, f[1]), 1))
        d21 = self.bcon2_1(torch.cat((d32, f[0]), 1))
        out = d21

        d43 = self.conv_d1(d43)
        d32 = torch.cat((d43, d32), dim=1)
        d32 = self.conv_d2(d32)
        d21 = torch.cat((d32, d21), dim=1)
        d21 = self.conv_d3(d21)

        return d21, out, d32, d43


class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()

        self.backbone_rgb = pvt_v2_b4(True)

        self.rgb_decoder = RGB_Decoder([64, 128, 320, 512])

        self.fm1 = FusionModule(64)
        self.fm2 = FusionModule(128)
        self.fm3 = FusionModule(320)
        self.fm4 = FusionModule(512)

        self.sm1 = SearchModule(64)
        self.sm2 = SearchModule(128)
        self.sm3 = SearchModule(320)
        self.sm4 = SearchModule(512)

        self.gt_out = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.edge_out = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.out1_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.out2_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.out3_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.out4_best = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 1, 3, 1, 1)
        )

        self.decoder = Decoder(64, 128, 320, 512)

    def forward(self, x, y):
        rgb = self.backbone_rgb(x)
        depth = self.backbone_rgb(y)

        rgb_ = rgb

        f1 = self.fm1(rgb[0], depth[0])
        f2 = self.fm2(rgb[1], depth[1])
        f3 = self.fm3(rgb[2], depth[2])
        f4 = self.fm4(rgb[3], depth[3])

        ff, fe = self.rgb_decoder(rgb_)

        s1 = self.sm1(f1, ff, fe)
        s2 = self.sm2(f2, ff, fe)
        s3 = self.sm3(f3, ff, fe)
        s4 = self.sm4(f4, ff, fe)
        features = [s1, s2, s3, s4]
        out = self.decoder(features)

        return (F.sigmoid(self.out1_best(out[0])), F.sigmoid(self.out2_best(out[1])), F.sigmoid(self.out3_best(out[2])), F.sigmoid(self.out4_best(out[3])),
                F.sigmoid(self.gt_out(ff)), F.sigmoid(self.edge_out(fe)))


if __name__ == '__main__':
    model = SNet().cuda()

    right = torch.randn(1, 3, 224, 224).cuda()
    left = torch.randn(1, 3, 224, 224).cuda()

    flops, params = profile(model, (right, left))
    print("flops:%.2f G，params:%.2f M" % (flops / 1e9, params / 1e6))

    out = model(right, left)
    for i in out:
        print(i.shape)