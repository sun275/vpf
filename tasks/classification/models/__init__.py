"""Classification model builder."""

from vpf.models import VPF, vpf


def build_vpf_model(config, is_pretrain=False):
    del is_pretrain
    if config.MODEL.TYPE != "vpf":
        raise ValueError(f"Unsupported model type: {config.MODEL.TYPE}")
    return VPF(
        in_chans=config.MODEL.VPF.IN_CHANS,
        patch_size=config.MODEL.VPF.PATCH_SIZE,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VPF.DEPTHS,
        dims=config.MODEL.VPF.EMBED_DIM,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        mlp_ratio=config.MODEL.VPF.MLP_RATIO,
        post_norm=config.MODEL.VPF.POST_NORM,
        layer_scale=config.MODEL.VPF.LAYER_SCALE,
        img_size=config.DATA.IMG_SIZE,
        infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,
        ablation=config.MODEL.VPF.ABLATION,
    )


def build_model(config, is_pretrain=False):
    return build_vpf_model(config, is_pretrain)


__all__ = ["VPF", "build_model", "build_vpf_model", "vpf"]
