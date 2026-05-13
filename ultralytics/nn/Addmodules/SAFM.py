import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data (used for fused inference)."""
        return self.act(self.conv(x))


class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # depthwise conv
        self.dwconv = self.dconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)
        return x6


class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)
        return x6


class FCM(nn.Module):
    """ Feature-Channel-Mixed module (kept mostly as-is, minor style tweaks)."""
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.dim = dim
        self.one = dim // 4
        self.two = dim - self.one
        # small conv path on split
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        # the other branch
        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim_out, 1, 1)
        # attentions
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)  # -> [B, dim, H, W]
        x4 = self.conv2(x2)    # -> [B, dim, H, W]
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        x5 = self.conv3(x5)
        return x5  # [B, dim_out, H, W]


class DeformConv2dPure(nn.Module):
    """
    Pure PyTorch implementation of Deformable Convolution (research/experiment).
    Not optimized for speed. Expects offset shape [B, 2*K, H_out, W_out].
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # convolution weight param: [out_channels, in_channels, k, k]
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, offset):
        """
        x: [B, C, H, W]
        offset: [B, 2*K, H_out, W_out], K = kernel_size^2
        """
        B, C, H, W = x.shape
        K = self.kernel_size * self.kernel_size

        # compute output spatial dims
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # check offset shape
        if offset.shape[2] != H_out or offset.shape[3] != W_out:
            # try to interpolate offsets to correct size (safer than failing)
            offset = F.interpolate(offset, size=(H_out, W_out), mode='bilinear', align_corners=True)

        # padding the input
        x_pad = F.pad(x, [self.padding] * 4)  # left,right,top,bottom

        # create base grid for kernel centers (before adding offset)
        # ys, xs are the top-left sampling locations *stride to get center positions
        ys = (torch.arange(H_out, device=x.device, dtype=x.dtype) * self.stride)
        xs = (torch.arange(W_out, device=x.device, dtype=x.dtype) * self.stride)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')  # H_out x W_out

        sampled_list = []
        # loop over kernel positions
        for k in range(K):
            dx = offset[:, k * 2, :, :].to(x_pad.dtype)  # [B, H_out, W_out]
            dy = offset[:, k * 2 + 1, :, :].to(x_pad.dtype)

            # grid coords for this kernel pos: shape [B, H_out, W_out]
            grid_x = xs.unsqueeze(0).expand(B, -1, -1) + dx
            grid_y = ys.unsqueeze(0).expand(B, -1, -1) + dy

            # normalize to [-1, 1] relative to padded image size (W_pad, H_pad)
            H_pad = x_pad.shape[2]
            W_pad = x_pad.shape[3]
            grid_x_norm = 2.0 * grid_x / (W_pad - 1) - 1.0
            grid_y_norm = 2.0 * grid_y / (H_pad - 1) - 1.0

            grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1)  # [B, H_out, W_out, 2]
            # sample
            sampled_k = F.grid_sample(x_pad, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_list.append(sampled_k)

        # concat K sampled maps -> [B, C*K, H_out, W_out]
        sampled = torch.cat(sampled_list, dim=1)

        # reshape weight to conv over C*K channels: weight_flat shape [out, in*K, 1,1]
        weight_flat = self.weight.view(self.out_channels, -1, 1, 1).to(sampled.dtype)

        out = F.conv2d(sampled, weight_flat, bias=self.bias)
        return out


class AlignConv(nn.Module):
    """
    RT-DETR style AlignConv with offset estimation + deform conv.
    Offset conv uses the same stride as deform conv so that offset spatial dims match.
    """
    def __init__(self, in_channels, out_channels=None,
                 kernel_size=3, stride=1, padding=1,
                 use_bn=True, use_relu=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # offset conv outputs 2*K channels. Use stride equal to deform conv stride so sizes match.
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )
        self.dcn = DeformConv2dPure(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # offset shape -> [B, 2*K, H_out, W_out] because offset_conv uses same stride
        offset = self.offset_conv(x)
        out = self.dcn(x, offset)
        if self.use_bn:
            out = self.bn(out)
        if self.use_relu:
            out = self.relu(out)
        return out


class SAFM(nn.Module):

    def __init__(self, in_channels, out_channels=None,
                 kernel_size=3, stride=1, padding=1,
                 use_bn=True, use_relu=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # semantic enhancement path
        self.fcm = FCM(dim=in_channels, dim_out=in_channels)

        # alignment path
        self.align = AlignConv(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               use_bn=use_bn, use_relu=use_relu)

        # fusion conv: reduce channels back to out_channels
        self.fuse_conv = Conv(in_channels * 2, out_channels, k=1, s=1, p=0)
        # optional residual projection if in != out
        if in_channels != out_channels:
            self.proj = Conv(in_channels, out_channels, k=1, s=1, p=0)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        # semantic enhanced features
        f_sem = self.fcm(x)  # [B, C, H, W]
        # alignment features (geometric alignment)
        f_align = self.align(x)  # [B, C, H_out, W_out] ; usually H_out==H if stride==1

        # if stride reduced spatial dims, upsample align to f_sem's spatial size for concat
        if f_align.shape[2:] != f_sem.shape[2:]:
            f_align = F.interpolate(f_align, size=f_sem.shape[2:], mode='bilinear', align_corners=True)

        # concat and fuse
        fused = torch.cat([f_sem, f_align], dim=1)  # [B, 2C, H, W]
        out = self.fuse_conv(fused)  # [B, out_channels, H, W]

        # add residual projection from input
        res = self.proj(x)
        out = out + res
        return out



