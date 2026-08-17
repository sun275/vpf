"""Shared OpenMMLab backbone implementation."""

from pathlib import Path

import torch
from mmengine.model import BaseModule
from torch import nn

from vpf.models import VPF


class OpenMMLabVPF(BaseModule, VPF):
    """Dense-prediction wrapper around the VPF classifier backbone."""

    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=(2, 2, 9, 2),
        dims=(96, 192, 384, 768),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        post_norm=True,
        layer_scale=None,
        use_checkpoint=False,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        img_size=224,
        **kwargs,
    ):
        del drop_rate, attn_drop_rate, norm_layer
        BaseModule.__init__(self)
        VPF.__init__(
            self,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            post_norm=post_norm,
            layer_scale=layer_scale,
            img_size=img_size,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.out_indices = tuple(out_indices)
        for index in self.out_indices:
            self.add_module(f"outnorm{index}", nn.LayerNorm(self.dims[index]))
        del self.classifier

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, checkpoint):
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VPF checkpoint does not exist: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            state_dict = payload.get("model_ema", payload.get("model", payload))
        else:
            state_dict = payload
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        return self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.patch_embed(x)
        outputs = []
        for index, layer in enumerate(self.layers):
            stage_output, x = layer(x, return_before_downsample=True)
            if index in self.out_indices:
                norm = getattr(self, f"outnorm{index}")
                output = norm(stage_output.permute(0, 2, 3, 1))
                outputs.append(output.permute(0, 3, 1, 2).contiguous())
        return tuple(outputs)
