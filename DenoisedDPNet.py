import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from thop import profile
from Mine_Net.mix_transformer import mit_b4
from torch.nn.functional import interpolate


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

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


class ImprovedGraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ImprovedGraphConvLayer, self).__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        # Enhanced graph convolution with normalization and non-linearity
        support = torch.matmul(adj, x)
        output = self.projection(support)
        output = self.gelu(output)
        output = self.norm(output)
        return output


class EnhancedDomainSeparationModule(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        self.dim = dim
        self.reduced_dim = dim // reduction_ratio

        # Enhanced feature reduction with residual connection
        self.reducer = nn.Sequential(
            nn.Conv2d(dim, self.reduced_dim, 1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.GELU(),
            nn.Conv2d(self.reduced_dim, self.reduced_dim, 3, padding=1, groups=self.reduced_dim),
            nn.BatchNorm2d(self.reduced_dim),
            nn.GELU()
        )

        # Improved graph convolution layers
        self.gcn1 = ImprovedGraphConvLayer(self.reduced_dim, self.reduced_dim)
        self.gcn2 = ImprovedGraphConvLayer(self.reduced_dim, self.reduced_dim)
        self.gcn3 = ImprovedGraphConvLayer(self.reduced_dim, self.reduced_dim)  # Additional GCN layer

        # Enhanced domain separator with skip connections
        self.domain_separator = nn.Sequential(
            nn.Conv2d(self.reduced_dim, self.reduced_dim * 2, 1),
            nn.BatchNorm2d(self.reduced_dim * 2),
            nn.GELU(),
            nn.Conv2d(self.reduced_dim * 2, self.reduced_dim * 2, 3, padding=1),
            nn.BatchNorm2d(self.reduced_dim * 2),
            nn.GELU(),
            nn.Conv2d(self.reduced_dim * 2, self.reduced_dim * 2, 1)
        )

        # Adaptive feature selection gate
        self.feature_gate = nn.Sequential(
            nn.Conv2d(self.reduced_dim * 2, self.reduced_dim, 1),
            nn.BatchNorm2d(self.reduced_dim),
            nn.GELU(),
            nn.Conv2d(self.reduced_dim, 2, 1),
            nn.Softmax(dim=1)
        )

        # Enhanced feature rebuilder with residual connection
        self.rebuilder = nn.Sequential(
            nn.Conv2d(self.reduced_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # Multi-scale channel attention
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(dim, dim // 16, 1),
                nn.GELU(),
                nn.Conv2d(dim // 16, dim, 1),
                nn.Sigmoid()
            ) for size in [1, 2, 4]
        ])

        # Enhanced spatial attention with multi-scale processing
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=7, padding=3),
            nn.BatchNorm2d(2),
            nn.GELU(),
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def construct_graph(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # B, C, HW

        # similarity computation with temperature scaling
        sim_matrix = torch.matmul(x_flat.transpose(1, 2), x_flat)  # B, HW, HW
        temperature = torch.sqrt(torch.tensor(C).float())
        sim_matrix = F.softmax(sim_matrix / temperature, dim=-1)

        # Add self-loops to prevent information loss
        identity = torch.eye(sim_matrix.size(1), device=sim_matrix.device)
        identity = identity.unsqueeze(0).expand(B, -1, -1)
        sim_matrix = sim_matrix + 0.1 * identity

        return F.normalize(sim_matrix, p=1, dim=2)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # 1. feature reduction
        x_reduced = self.reducer(x)
        reduced_residual = x_reduced  # Save for skip connection

        # 2. Enhanced1
        adj_matrix = self.construct_graph(x_reduced)

        x_flat = x_reduced.view(B, self.reduced_dim, -1).transpose(1, 2)
        x_gcn = self.gcn1(x_flat, adj_matrix)
        x_gcn = self.gcn2(x_gcn, adj_matrix)
        x_gcn = self.gcn3(x_gcn, adj_matrix)  # Additional GCN processing
        x_gcn = x_gcn.transpose(1, 2).view(B, self.reduced_dim, H, W)

        x_gcn = x_gcn + reduced_residual

        # 3. domain separation
        domain_features = self.domain_separator(x_gcn)
        salient_domain, noise_domain = torch.chunk(domain_features, 2, dim=1)

        gates = self.feature_gate(domain_features)
        salient_gate, noise_gate = gates[:, 0:1, :, :], gates[:, 1:2, :, :]

        # 4. feature fusion with learnable weights
        refined_features = salient_gate * salient_domain - noise_gate * noise_domain
        output = self.rebuilder(refined_features)

        # 5. Enhancement2
        ca_outputs = []
        for ca_module in self.channel_attention:
            ca_out = ca_module(output)
            if ca_out.size(-1) != H:
                ca_out = F.interpolate(ca_out, size=(H, W), mode='bilinear', align_corners=True)
            ca_outputs.append(ca_out)

        ca = sum(ca_outputs) / len(ca_outputs)
        output = output * ca

        # 6. spatial attention with multi-scale features
        avg_pool = torch.mean(output, dim=1, keepdim=True)
        max_pool, _ = torch.max(output, dim=1, keepdim=True)
        min_pool, _ = torch.min(output, dim=1, keepdim=True)
        std_pool = torch.std(output, dim=1, keepdim=True)
        spatial = torch.cat([avg_pool, max_pool, min_pool, std_pool], dim=1)
        spatial = self.spatial_attention(spatial)
        output = output * spatial

        # Residual connection with adaptive weighting
        output = output + residual

        return output


class DomainSeparationBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.separators = nn.ModuleList([
            EnhancedDomainSeparationModule(dim) for dim in dims
        ])

    def forward(self, features):
        outputs = []
        for feat, separator in zip(features, self.separators):
            outputs.append(separator(feat))
        return outputs


class CleaningModule(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        self.dim = dim
        self.reduced_dim = dim // reduction_ratio

        # Dimension reduction
        self.reduce = nn.Conv2d(dim, self.reduced_dim, 1)

        # Efficient attention mechanism
        self.q = nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        self.k = nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        self.v = nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)

        # Restoration
        self.restore = nn.Conv2d(self.reduced_dim, dim, 1)

        # Denoising components
        self.denoise = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Noise estimation (simplified)
        self.noise_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        residual = x

        # Reduce dimension
        x_reduced = self.reduce(x)

        # Efficient attention
        B, C, H, W = x_reduced.shape
        q = self.q(x_reduced).view(B, self.reduced_dim, -1)
        k = self.k(x_reduced).view(B, self.reduced_dim, -1)
        v = self.v(x_reduced).view(B, self.reduced_dim, -1)

        attn = (q.transpose(1, 2) @ k) * (1.0 /- (H * W))
        attn = attn.softmax(dim=-1)

        x_attn = (v @ attn.transpose(1, 2)).view(B, self.reduced_dim, H, W)

        # Restore dimension
        x = self.restore(x_attn)

        # Denoising
        x_denoised = self.denoise(x)

        # Noise estimation and removal
        noise = self.noise_estimator(x)
        x = x - noise

        # Combine with residual
        x = x + residual + x_denoised

        return x


class EnhancementModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False):
        super(EnhancementModule, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_planes))
        else:
            self.register_parameter('bias', None)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.attention(x)
        batch_size, _, height, width = x.size()

        weight = self.weight.view(self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)


        # Change this line
        aggregate_weight = weight.view(self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = self.bias.repeat(batch_size)
        else:
            aggregate_bias = None

        # Modify this line Cha
        # output = F.conv2d(x + attention, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
        #                   groups=1)

        output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                          groups=1)

        return output * attention


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
        f3 = self.bcon4(f[3])
        f2 = self.bcon3(f[2])
        f1 = self.bcon2(f[1])
        f0 = self.bcon1(f[0])

        d43 = self.bcon4_3(torch.cat((f3, f2), 1))
        d32 = self.bcon3_2(torch.cat((d43, f1), 1))
        d21 = self.bcon2_1(torch.cat((d32, f0), 1))
        out = d21

        d43 = self.conv_d1(d43)
        d32 = torch.cat((d43, d32), dim=1)
        d32 = self.conv_d2(d32)
        d21 = torch.cat((d32, d21), dim=1)
        d21 = self.conv_d3(d21)

        return d21, out, d32, d43


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

        # Keep the original backbone
        self.backbone_rgb = mit_b4()
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(
            '/media/tbb/9b281502-670b-4aec-957e-085adc101020/UAV/USOD/Backbone_pth/segformer.b4.512x512.ade.160k.pth')[
            'state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backbone_rgb.load_state_dict(new_state_dict3, strict=False)

        # Enhanced cleaning modules
        self.ssdtm = nn.ModuleList([CleaningModule(64), CleaningModule(128),
                                    CleaningModule(320), CleaningModule(512)])

        # Keep the original decoder
        self.decoder = Decoder(64, 128, 320, 512)

        # Enhanced dynamic convolution
        self.dynamic_conv = nn.ModuleList([
            EnhancementModule(64, 64, 3, padding=1),
            EnhancementModule(128, 128, 3, padding=1),
            EnhancementModule(320, 320, 3, padding=1),
            EnhancementModule(512, 512, 3, padding=1)
        ])

        # Enhanced domain separator
        self.domain_separator_rgb = DomainSeparationBlock([64, 128, 320, 512])
        self.domain_separator_depth = DomainSeparationBlock([64, 128, 320, 512])

        # Keep the original output layers
        self.last1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.last2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.last3 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.last4 = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 1, 3, 1, 1)
        )

    def forward(self, x, y):
        # Extract features from RGB and depth streams
        rgb = self.backbone_rgb(x)
        depth = self.backbone_rgb(y)

        # Enhanced domain separation

        rgb = self.domain_separator_rgb(rgb)
        depth = self.domain_separator_depth(depth)

        # Improved feature fusion
        merges = []
        for i in range(4):
            # Enhanced feature cleaning and fusion
            rgb[i] = self.ssdtm[i](rgb[i]+depth[i])
            fused = self.dynamic_conv[i](rgb[i])
            merges.append(fused)

        # Keep the original decoder path
        out = self.decoder(merges)

        # Ensure consistent output sizes
        out0 = self.last1(out[0])
        out1 = self.last2(out[1])
        out2 = self.last3(out[2])
        out3 = self.last4(out[3])

        return F.sigmoid(out0), F.sigmoid(out1), F.sigmoid(out2), F.sigmoid(out3)

if __name__ == '__main__':
    model = DNet().cuda()

    right = torch.randn(8, 3, 224, 224).cuda()
    left = torch.randn(8, 3, 224, 224).cuda()

    flops, params = profile(model, (right, left ))
    print("flops:%.2f G，params:%.2f M" % (flops / 1e9, params / 1e6))

    out = model(right, left)
    for i in out:
        print(i.shape)