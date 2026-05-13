import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    def __init__(self, lambda_val: float = 1e-4):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        # x: [B,C,H,W]
        b, c, h, w = x.size()
        n = h * w - 1 if h * w > 1 else 1
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_var = ((x - x_mean) ** 2).sum(dim=(2, 3), keepdim=True) / n
        e_inv = (x - x_mean) ** 2 / (4 * (x_var + self.lambda_val)) + 0.5  # 能量反比
        attn = torch.sigmoid(e_inv)
        return x * attn

class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        x_h = self.pool_h(x)               # [B,C,H,1]
        x_w = self.pool_w(x).permute(0,1,3,2)  # [B,C,1,W] -> [B,C,W,1]

        y = torch.cat([x_h, x_w], dim=2)   # [B,C,H+W,1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0,1,3,2)

        a_h = self.sigmoid(self.conv_h(y_h))   # [B,C,H,1]
        a_w = self.sigmoid(self.conv_w(y_w))   # [B,C,1,W]
        out = identity * a_h * a_w
        return out


class LightEMA(nn.Module):
    def __init__(self, channels, kernels=(3, 5, 7), groups=None):
        super().__init__()
        if groups is None:
            # 让 groups 尽量整除，默认 = 通道数（纯 depthwise）
            groups = channels
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k//2, groups=groups, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for k in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (len(kernels) + 1), channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.scale = nn.Parameter(torch.tensor(0.25))  # 温和融合

    def forward(self, x):
        feats = [x]
        for br in self.branches:
            feats.append(br(x))
        y = self.fuse(torch.cat(feats, dim=1))
        return x + y * self.scale


class EdgeExtraction(nn.Module):
    
    def __init__(self, in_channels):
        super(EdgeExtraction, self).__init__()
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.edge_scale = nn.Parameter(torch.tensor(0.25))  # 更保守的初值
        self._init_sobel_weights()

    def _init_sobel_weights(self):
        device = self.sobel_x.weight.device
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device)
        kx = kx.view(1, 1, 3, 3)
        ky = ky.view(1, 1, 3, 3)
        kx = kx.repeat(self.sobel_x.weight.size(0), 1, 1, 1)
        ky = ky.repeat(self.sobel_y.weight.size(0), 1, 1, 1)
        with torch.no_grad():
            self.sobel_x.weight.copy_(kx)
            self.sobel_y.weight.copy_(ky)

        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edges * self.edge_scale


class FeatureEnhancement(nn.Module):
    
    def __init__(self, channels):
        super(FeatureEnhancement, self).__init__()
        mid = max(8, channels // 2)
        self.residual_block = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.scale = nn.Parameter(torch.tensor(0.2))  # 残差缩放

    def forward(self, x):
        return x + self.residual_block(x) * self.scale



class FMEM(nn.Module):

    def __init__(self, in_channels, reduction=8, kernel_size=5):
        super(FMEM, self).__init__()

        
        self.blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  
        self.laplace = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        with torch.no_grad():
            lap = torch.tensor([[0., -1., 0.],
                                [-1., 4., -1.],
                                [0., -1., 0.]]).view(1,1,3,3)
            lap = lap.repeat(in_channels, 1, 1, 1)
            self.laplace.weight.copy_(lap)
        for p in self.laplace.parameters():
            p.requires_grad = False

        self.edge_extraction = EdgeExtraction(in_channels)
        self.feature_enhancement = FeatureEnhancement(in_channels)

        
        self.simam = SimAM(lambda_val=1e-4)
        self.coord_att = CoordAtt(in_channels, reduction=max(8, reduction))  
        self.ema = LightEMA(in_channels, kernels=(3, 5, 7), groups=in_channels)

        
        self.fuse_att = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        
        self.final_scale = nn.Parameter(torch.tensor(0.35))  

        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1)
        )

    def _foreground_mask(self, x):
        
        low = self.blur(x)
        contrast = torch.abs(x - low)         # 局部对比
        sharp = torch.relu(self.laplace(x))   # 高频/边界
        mix = contrast + 0.5 * sharp
        
        m = mix / (mix.amax(dim=(2,3), keepdim=True) + 1e-6)
        return torch.sigmoid(4.0 * (m - 0.5))  

    def forward(self, x):
        residual = x

        
        fg_mask = self._foreground_mask(x)
        x_fg = x * (0.5 + 0.5 * fg_mask)  # 背景不清零，避免信息丢失

        
        edge_feats = self.edge_extraction(x_fg)
        enhanced = self.feature_enhancement(x_fg + edge_feats)

        
        a1 = self.simam(enhanced)
        a2 = self.coord_att(enhanced)
        a3 = self.ema(enhanced)

        att = self.fuse_att(torch.cat([a1, a2, a3], dim=1))

       
        max_pool = torch.max(att, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(att, dim=1, keepdim=True)
        spa = self.spatial_gate(torch.cat([max_pool, avg_pool], dim=1))
        att = att * torch.sigmoid(spa) * 0.8  

        
        out = residual + att * self.final_scale
        return out
