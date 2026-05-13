import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


def get_activation(activation: str):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    else:
        raise ValueError(f"Unsupported activation: {activation}, only 'relu' and 'gelu' are allowed")


class GCBlock(nn.Module):
    """Global Context Block (lightweight global context / simplified non-local)"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        context = self.pool(x)
        transformed = self.transform(context)
        gate = self.sigmoid(transformed)
        out = x * gate + x
        return out


class DepthwiseDilatedConv(nn.Module):
    """Depthwise dilated conv to enlarge receptive field with tiny cost"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1,
                            padding=padding, dilation=dilation, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        return out


class APE_LiteLayer(nn.Module):
    
    def __init__(self, channels: int, dim_feedforward: int = 1024, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        hidden = min(dim_feedforward, max(64, channels // 2))
        act = get_activation(activation)

        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.dw = DepthwiseDilatedConv(hidden, kernel_size=3, dilation=2)
        self.gc = GCBlock(hidden, reduction=8)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm([channels, 1, 1])
        self.activation = act

    def forward(self, x, pos_add: torch.Tensor = None):
        
        dtype = self.reduce.weight.dtype
        if x.dtype != dtype:
            x = x.to(dtype)
        if pos_add is not None and pos_add.dtype != dtype:
            pos_add = pos_add.to(dtype)
            x = x + pos_add

        residual = x
        x = self.reduce(x)
        x = self.dw(x)
        x = self.gc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.expand(x)

        out = residual + x

        B, C, H, W = out.shape
        out_ln = out.permute(0, 2, 3, 1).contiguous()
        out_ln = F.layer_norm(out_ln, (C,))
        out = out_ln.permute(0, 3, 1, 2).contiguous()
        return out


class GPEM(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        window_size: int,
        nhead: int = 8,
        num_layers: int = 3,#默认1
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
        norm: nn.Module = None,
        use_pos_embed: bool = True
    ):
        super().__init__()
        self.window_size = window_size
        self.num_layers = num_layers
        self.use_pos_embed = use_pos_embed
        self.layers = nn.ModuleList([
            APE_LiteLayer(d_model, dim_feedforward=dim_feedforward, activation=activation, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = norm

    @staticmethod
    def build_sine_position_embedding(C, H, W, device='cpu', dtype=torch.float32):
        pe = torch.zeros(1, C, H, W, device=device, dtype=dtype)
        y_pos = torch.arange(0, H, device=device).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(0, W, device=device).unsqueeze(0).repeat(H, 1)
        div_term = torch.exp(torch.arange(0, C, 2, device=device, dtype=dtype) * -(torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / C))
        for i in range(0, C, 2):
            pe[0, i, :, :] = torch.sin(x_pos * div_term[i // 2])
            if i + 1 < C:
                pe[0, i + 1, :, :] = torch.cos(y_pos * div_term[i // 2])
        return pe

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None,
                pos_embed: torch.Tensor = None, glob_pos_embeds: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = src.shape

        
        if self.use_pos_embed and pos_embed is None and glob_pos_embeds is None:
            pos_embed = self.build_sine_position_embedding(C, H, W, device=src.device, dtype=src.dtype)

        # prepare positional addition tensor
        pos_add = None
        if glob_pos_embeds is not None and pos_embed is not None:
            g = glob_pos_embeds
            p = pos_embed
            if g.dim() == 3: g = g.unsqueeze(0).expand(B, -1, -1, -1)
            if p.dim() == 3: p = p.unsqueeze(0).expand(B, -1, -1, -1)
            pos_add = g + p
        elif glob_pos_embeds is not None:
            pos_add = glob_pos_embeds if glob_pos_embeds.dim() == 4 else glob_pos_embeds.unsqueeze(0).expand(B, -1, -1, -1)
        elif pos_embed is not None:
            pos_add = pos_embed if pos_embed.dim() == 4 else pos_embed.unsqueeze(0).expand(B, -1, -1, -1)

        
        if src.dtype != self.layers[0].reduce.weight.dtype:
            src = src.to(self.layers[0].reduce.weight.dtype)
        if pos_add is not None and pos_add.dtype != src.dtype:
            pos_add = pos_add.to(src.dtype)

        out = src
        for layer in self.layers:
            out = layer(out, pos_add)

        if self.norm is not None:
            try:
                out = self.norm(out)
            except Exception:
                B, C, H, W = out.shape
                out_perm = out.permute(0, 2, 3, 1).contiguous()
                out_perm = F.layer_norm(out_perm, (C,))
                out = out_perm.permute(0, 3, 1, 2).contiguous()

        return out
