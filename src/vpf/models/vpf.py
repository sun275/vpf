import time
import math
from functools import partial
from typing import Optional, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()
    
    
def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def check(x, name):
    # 检查是否包含 NaN 或 Inf
    if torch.isnan(x).any():
        print(f"[NaN detected after {name}]")
    elif torch.isinf(x).any():
        print(f"[Inf detected after {name}]")


class phy_field(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode

        self.raw_tau = nn.Parameter(torch.full((1, 1, 1, hidden_dim), math.log(math.e - 1.0)))
        self.raw_rho = nn.Parameter(torch.full((1, 1, 1, hidden_dim), -10.0))
        self.rho_max = 0.2

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        k2 = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        return torch.exp(-k2), k2

    def forward(self, x, param, compute_spectral_state=True):
        # check(x, "before phy_field")
        B, C, H, W = x.shape

        # check(x, "before dwconv")
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            k2 = getattr(self, "__K2_MAP__", None)
            low_freq_mask = getattr(self, "__LOW_FREQ_MASK__", None)
            high_freq_mask = getattr(self, "__HIGH_FREQ_MASK__", None)

            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
            assert k2 is not None
            assert low_freq_mask is not None
            assert high_freq_mask is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp, k2 = self.get_decay_map((H, W), device=x.device)
            weight_exp = weight_exp.detach_()
            k2 = k2.detach_()
            low_freq_mask = k2 <= (0.5 * k2[-1, -1])
            high_freq_mask = ~low_freq_mask
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)
            setattr(self, "__K2_MAP__", k2)
            setattr(self, "__LOW_FREQ_MASK__", low_freq_mask)
            setattr(self, "__HIGH_FREQ_MASK__", high_freq_mask)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        # check(x, "before DCT")
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        kernel_cache_key = (
            H, W, x.device, x.dtype, self.raw_tau._version, self.raw_rho._version
        )
        if self.infer_mode and kernel_cache_key == getattr(self, "__KERNEL_CACHE_KEY__", None):
            kernel = getattr(self, "__KERNEL__", None)
            assert kernel is not None
        else:
            tau = F.softplus(self.raw_tau).clamp(max=4.0)
            rho = torch.sigmoid(self.raw_rho) * self.rho_max
            weight_exp = weight_exp.clamp_min(1e-6)
            kernel = rho + (1.0 - rho) * torch.exp(
                tau * torch.log(weight_exp[None, :, :, None])
            )
            if self.infer_mode:
                setattr(self, "__KERNEL_CACHE_KEY__", kernel_cache_key)
                setattr(self, "__KERNEL__", kernel.detach())

        effective_mask = kernel * param
        spectral_state = None
        if compute_spectral_state:
            low_weights = low_freq_mask[None, :, :, None].to(effective_mask.dtype)
            high_weights = high_freq_mask[None, :, :, None].to(effective_mask.dtype)
            low_score = (effective_mask * low_weights).sum(dim=(1, 2)) / low_weights.sum().clamp_min(1.0)
            high_score = (effective_mask * high_weights).sum(dim=(1, 2)) / high_weights.sum().clamp_min(1.0)
            spectral_state = torch.cat([low_score, high_score], dim=-1)

        # check(x, "before phy_field.mul_param")
        x = x * effective_mask
        # check(x, "phy_field.mul_param")

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        # check(x, "after phy_field")
        return x, spectral_state
    

class DNO2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, use_dynamic_param=True):
#         super().__init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs)
        super().__init__()
        self.use_dynamic_param = use_dynamic_param
        if self.use_dynamic_param:
            self.param_norm = nn.LayerNorm(hidden_dim)
            self.dwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
            self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.solve = phy_field(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.apply(self._default_init)

    def _default_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # =========================================================
    # [核心方法] 加载物理先验权重并初始化 Driver
    # =========================================================
    def load_and_init_physics(self, weight_path):
        """
        weight_path: 对应当前 dim 的 .pth 文件路径
        """
        # 1. 加载 phy_field (引擎) 的预训练权重
        # -----------------------------------------------------
        if os.path.exists(weight_path):
            try:
                # 加载权重到 cpu
                state_dict = torch.load(weight_path, map_location='cpu')
                
                # 加载给 self.solve
                # strict=False 只要 key 匹配即可，忽略一些缓存 buffer 的差异
                msg = self.load_state_dict(state_dict, strict=False)
                print(f"  [DNO2D] Successfully loaded FULL weights from: {os.path.basename(weight_path)}")
            except Exception as e:
                print(f"  [Error] Failed to load {weight_path}: {e}")
        else:
            print(f"  [Warning] Physics weight not found: {weight_path}")
    def forward(self, x, compute_spectral_state=True):
        if self.use_dynamic_param:
            param = self.pointwise(self.dwconv1(x))
            param = param.permute(0,2,3,1)
            param = self.param_norm(param)
            param = torch.sigmoid(param)
        else:
            param = x.new_ones((x.shape[0], x.shape[2], x.shape[3], 1))
        return self.solve(x, param, compute_spectral_state=compute_spectral_state)

class galerkin_attn(nn.Module):
    def __init__(self, dim, heads, inner_dim=None):
        super().__init__()

        inner_dim = inner_dim or dim
        assert inner_dim % heads == 0
        self.headc = inner_dim // heads
        self.heads = heads
        self.inner_dim = inner_dim

        self.in_proj = nn.Conv2d(dim, inner_dim, 1)
        self.ln1 = nn.GroupNorm(1, inner_dim)
        self.qkv_proj = nn.Conv2d(inner_dim, 3 * inner_dim, 1)
        self.out_proj = nn.Conv2d(inner_dim, inner_dim, 1)
        self.expand_proj = nn.Conv2d(inner_dim, dim, 1)

        self.kln = nn.LayerNorm(self.headc, eps=1e-5)
        self.vln = nn.LayerNorm(self.headc, eps=1e-5)
        self.qln = nn.LayerNorm(self.headc, eps=1e-5)
    
    def forward(self, x):
        # check(x,"before galerkin")
        B, _, H, W = x.shape
        # check(x,"before PROJ")

        x_norm = self.ln1(self.in_proj(x))
        qkv = self.qkv_proj(x_norm).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)

        qkv = qkv.permute(0, 2, 1, 3)
        # check(qkv,"AFTER PROJ")

        q, k, v = qkv.chunk(3, dim=-1)
        # check(v,"before calv")


        k = self.kln(k)
        v = self.vln(v)
        q = self.qln(q)
        scale = 1.0 / (H * W)
        # check(v,"before torch.matmul(k.transpose(-2,-1), v)")

        # global_context = torch.matmul(k.transpose(-2,-1).float(), v.float())*scale
        # # 转回原来的精度 (如 fp16) 继续后面的计算
        # global_context = global_context.to(q.dtype)
        global_context = torch.matmul(k.transpose(-2,-1), v)*scale
        # check(global_context,"after torch.matmul(k.transpose(-2,-1), v)")

        v = torch.matmul(q, global_context)

        v = v.permute(0, 2, 1, 3).reshape(B, H, W, self.inner_dim)
        # check(v,"after calv")

        global_delta = self.out_proj(v.permute(0, 3, 1, 2).contiguous())
        global_delta = self.expand_proj(global_delta)
        # check(global_delta,"after galerkin")
        return global_delta


class DNOBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode = False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm = True,
        layer_scale = None,
        use_dno = True,
        # heads =8,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = DNO2D(
            res=res,
            dim=hidden_dim,
            hidden_dim=hidden_dim,
            infer_mode=infer_mode,
            use_dynamic_param=use_dno,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        
        self.infer_mode = infer_mode
        
        # self.AttenNO = galerkin_attn(hidden_dim,heads=heads)
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
        
        # self.gateconv = nn.Conv2d(hidden_dim*2, 1, 1)
    def _forward(self, x: torch.Tensor, compute_spectral_state=True):
        # check(x,"before block")
        if not self.layer_scale:
            if self.post_norm:
                op_out, spectral_state = self.op(x, compute_spectral_state)
                x = x + self.drop_path(self.norm1(op_out))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                op_out, spectral_state = self.op(self.norm1(x), compute_spectral_state)
                x = x + self.drop_path(op_out)
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.post_norm:
                op_out, spectral_state = self.op(x, compute_spectral_state)
                x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(op_out))
                if self.mlp_branch:
                    x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
            else:
                op_out, spectral_state = self.op(self.norm1(x), compute_spectral_state)
                x = x + self.drop_path(self.gamma1[:, None, None] * op_out)
                if self.mlp_branch:
                    x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))
        return x, spectral_state
    
    def forward(self, input: torch.Tensor, compute_spectral_state=True):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, compute_spectral_state)
        else:
            return self._forward(input, compute_spectral_state)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class DNOlayer(nn.Module):
    def __init__( self,res=14,
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(), 
        mlp_ratio=4.0,
        infer_mode=False,
        heads=8,
        use_galerkin=False,
        use_gate=True,
        use_dno=True,
        **kwargs):
        super().__init__()
        assert depth == len(drop_path)
        self.depth = depth
        blocks = []
        for d in range(depth):
            blocks.append(DNOBlock(
                res=res,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
                use_dno=use_dno,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.downsample = downsample
        self.use_galerkin = use_galerkin
        self.use_gate = use_gate
        if self.use_galerkin:
            self.global_branch = galerkin_attn(dim, heads=heads, inner_dim=dim // 2)
            self.global_alpha = nn.Parameter(torch.full((dim,), 1e-3))
            if self.use_gate:
                gate_hidden_dim = max(dim // 8, 1)
                self.feature_encoder = nn.Sequential(
                    nn.Linear(dim, gate_hidden_dim),
                    nn.GELU(),
                )
                self.spectral_encoder = nn.Sequential(
                    nn.Linear(2 * dim, gate_hidden_dim),
                    nn.GELU(),
                )
                self.joint_gate = nn.Sequential(
                    nn.Linear(2 * gate_hidden_dim, dim),
                    nn.Sigmoid(),
                )
    
    def forward(self, x, return_before_downsample=False):
        spectral_state = None
        last_block_index = len(self.blocks) - 1
        for block_index, block in enumerate(self.blocks):
            x, spectral_state = block(
                x,
                compute_spectral_state=self.use_galerkin and self.use_gate and block_index == last_block_index,
            )
        if self.use_galerkin:
            global_delta = self.global_branch(x)
            if self.use_gate:
                feature_state = x.mean(dim=(2, 3))
                feature_embed = self.feature_encoder(feature_state)
                spectral_embed = self.spectral_encoder(spectral_state)
                gate = self.joint_gate(torch.cat([feature_embed, spectral_embed], dim=-1))
                x = x + self.global_alpha[None, :, None, None] * gate[:, :, None, None] * global_delta
            else:
                x = x + self.global_alpha[None, :, None, None] * global_delta

        stage_output = x
        x = self.downsample(x)
        if return_before_downsample:
            return stage_output, x
        return x
    

class vpf(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=(2, 2, 17, 2),
                 dims=(96, 192, 384, 768), drop_path_rate=0.1, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False, pretrained_physics_dir=None,
                 ablation='full', **kwargs):
        super().__init__()
        valid_ablations = {'full', 'no_dno', 'no_galerkin', 'no_gate'}
        if ablation not in valid_ablations:
            raise ValueError(f"ablation must be one of {sorted(valid_ablations)}, got {ablation!r}")
        self.ablation = ablation
        use_dno = ablation != 'no_dno'
        use_galerkin = ablation != 'no_galerkin'
        use_gate = ablation != 'no_gate'
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        
        self.depths = depths
        
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=self.embed_dim,
                                     act_layer='GELU',
                                     norm_layer='LN')
        
        res0 = img_size/patch_size
        self.res = [int(res0), int(res0//2), int(res0//4), int(res0//8)]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.infer_mode = infer_mode
        
        # self.freq_embed = nn.ParameterList()
#         for i in range(self.num_layers):
#             self.freq_embed.append(nn.Parameter(torch.zeros(self.res[i], self.res[i], self.dims[i]), requires_grad=True))
#             trunc_normal_(self.freq_embed[i], std=.02)
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(DNOlayer(
                res = self.res[i_layer],
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=self.make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=LayerNorm2d,
                ) if (i_layer < self.num_layers - 1) else nn.Identity(),
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
                use_galerkin=use_galerkin,
                use_gate=use_gate,
                use_dno=use_dno,
            ))
            
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)
        # === [新增] 如果提供了路径，加载物理先验 ===
        if pretrained_physics_dir:
            self._load_physics_priors_globally(pretrained_physics_dir)
    def _load_physics_priors_globally(self, dir_path):
        print(f"\n>>> Initializing DNO with Diffusion Priors from: {dir_path}")
        
        # 遍历模型的 4 个 Stage
        for i, layer in enumerate(self.layers):
            current_dim = self.dims[i]
            
            # 预先定义该阶段需要的两种专家权重路径
            pos_weight_file = os.path.join(dir_path, f"dno_expert_positive_dim{current_dim}.pth")
            neg_weight_file = os.path.join(dir_path, f"dno_expert_negative_dim{current_dim}.pth")
            
            # 遍历当前 Stage 的所有 Block
            for b_idx, block in enumerate(layer.blocks):
                if hasattr(block, 'op') and isinstance(block.op, DNO2D):
                    # 核心逻辑：第一个 Block 加载正向，后续加载反向
                    if b_idx == 0:
                        target_weight = pos_weight_file
                        mode_label = "POSITIVE"
                    else:
                        target_weight = neg_weight_file
                        mode_label = "NEGATIVE"
                    
                    # 调用 DNO2D 自带的加载方法
                    block.op.load_and_init_physics(target_weight)
                    print(f"    Stage {i} | Block {b_idx}: Loaded {mode_label} (Dim {current_dim})")

        print(">>> Multi-mode Physics Initialization Complete.\n")

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'raw_tau', 'raw_rho', 'global_alpha'}

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            #norm_layer(dim),
            #nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )

    @staticmethod
    def make_layer(
        res=14,
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(), 
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(DNOBlock(
                
                res=res,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
            ))
        
        return AdditionalInputSequential(
            *blocks, 
            downsample,
        )
 
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear): 
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    


#     def infer_init(self):
#         for i, layer in enumerate(self.layers):
#             for block in layer[:-1]:
#                 block.op.infer_init_heat2d(self.freq_embed[i])
#         del self.freq_embed
    
    def forward_features(self, x):
        x = self.patch_embed(x)
#         if self.infer_mode:
        for layer in self.layers:
            x = layer(x)
#         else:
#             for i, layer in enumerate(self.layers):
#                 x = layer(x, self.freq_embed[i]) # (B, C, H, W)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
    model = vpf().cuda()
    input = torch.randn((1, 3, 224, 224), device=torch.device('cuda'))
    analyze = FlopCountAnalysis(model, input,)
    print(flop_count_str(analyze))
