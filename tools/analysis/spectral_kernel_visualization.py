#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


THIS_DIR = Path(__file__).resolve().parent
CLASSIFICATION_ROOT = THIS_DIR.parent
sys.path.insert(0, str(CLASSIFICATION_ROOT))

from vpf.models import phy_field, vpf  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_int_tuple(text):
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize learned DNO spectral kernels and effective masks."
    )
    parser.add_argument(
        "--checkpoint",
        default="/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth",
    )
    parser.add_argument("--data-root", default="/data1/sunmy/Imagenet1k/data/val")
    parser.add_argument(
        "--output-dir",
        default=str(CLASSIFICATION_ROOT / "analysis" / "spectral_kernel_outputs"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--depths", default="2,3,4,2")
    parser.add_argument("--dims", type=int, default=96)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--max-images",
        type=int,
        default=1024,
        help="Number of validation images used for dynamic effective-mask statistics.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_ema", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint


def strip_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = value
    return out


def build_model(args):
    model = vpf(
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=parse_int_tuple(args.depths),
        dims=args.dims,
        drop_path_rate=args.drop_path_rate,
        mlp_ratio=4.0,
        post_norm=True,
        img_size=args.img_size,
        infer_mode=False,
        ablation="full",
    )
    state_dict = strip_prefix(load_checkpoint(args.checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    model.to(args.device).eval()
    return model


def build_loader(args):
    transform = transforms.Compose(
        [
            transforms.Resize(int(args.img_size / 0.875)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def stage_from_name(name):
    parts = name.split(".")
    if "layers" not in parts:
        return None
    idx = parts.index("layers")
    if idx + 1 >= len(parts):
        return None
    try:
        return int(parts[idx + 1])
    except ValueError:
        return None


def band_masks(k2):
    k2_max = k2.max().clamp_min(1e-12)
    norm = k2 / k2_max
    return {
        "low": norm <= (1.0 / 3.0),
        "mid": (norm > (1.0 / 3.0)) & (norm <= (2.0 / 3.0)),
        "high": norm > (2.0 / 3.0),
    }


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, values):
        values = values.detach().float().cpu()
        self.count += values.numel()
        self.sum += values.sum().item()
        self.sumsq += (values * values).sum().item()

    @property
    def mean(self):
        return self.sum / max(self.count, 1)

    @property
    def std(self):
        if self.count <= 1:
            return 0.0
        var = max(self.sumsq / self.count - self.mean * self.mean, 0.0)
        return var ** 0.5


def normalized_mask(mask):
    mask = mask.astype(np.float32)
    max_value = float(mask.max())
    if max_value <= 1e-12:
        return mask
    return mask / max_value


def save_heatmap_grid(stage_maps, output_path, title, dpi):
    stages = sorted(stage_maps)
    fig, axes = plt.subplots(1, len(stages), figsize=(2.4 * len(stages), 2.35))
    if len(stages) == 1:
        axes = [axes]
    last_im = None
    for ax, stage in zip(axes, stages):
        image = normalized_mask(stage_maps[stage])
        last_im = ax.imshow(image, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"Stage {stage + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, y=1.03)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_band_plot(stats, output_path, dpi):
    stages = sorted({stage for stage, _ in stats})
    bands = ["low", "mid", "high"]
    colors = {"low": "#2166ac", "mid": "#4d9221", "high": "#b2182b"}
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    width = 0.24
    offsets = {"low": -width, "mid": 0.0, "high": width}
    for band in bands:
        means = np.array([stats[(stage, band)].mean for stage in stages])
        stds = np.array([stats[(stage, band)].std for stage in stages])
        stds = np.array([stats[(stage, band)].std for stage in stages])
        ax.bar(
            x + offsets[band],
            means,
            width=width,
            color=colors[band],
            alpha=0.78,
            label=f"{band}-frequency",
            yerr=stds,
            error_kw=dict(ecolor="0.25", elinewidth=1.0, capsize=2.5, capthick=1.0),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{stage + 1}" for stage in stages])
    ax.set_ylabel("Effective mask response")
    ax.set_title("Frequency-band response of DNO effective masks")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_csv(stats, output_path):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "band", "mean", "std", "count"])
        for (stage, band), stat in sorted(stats.items()):
            writer.writerow([stage + 1, band, stat.mean, stat.std, stat.count])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    loader = build_loader(args)

    effective_sums = defaultdict(lambda: None)
    effective_counts = defaultdict(int)
    kernel_sums = defaultdict(lambda: None)
    kernel_counts = defaultdict(int)
    band_stats = defaultdict(RunningStats)

    hooks = []

    def make_hook(stage, module):
        def hook(_module, inputs):
            x, param = inputs[:2]
            _, _, height, width = x.shape
            weight_exp, k2 = module.get_decay_map(
                (height, width), device=x.device, dtype=x.dtype
            )
            weight_exp = weight_exp.clamp_min(1e-6)
            tau = F.softplus(module.raw_tau).clamp(max=4.0)
            rho = torch.sigmoid(module.raw_rho) * module.rho_max
            kernel = rho + (1.0 - rho) * torch.exp(
                tau * torch.log(weight_exp[None, :, :, None])
            )
            effective = kernel * param

            kernel_sum = kernel.detach().float().sum(dim=(0, 3)).cpu().numpy()
            effective_sum = effective.detach().float().sum(dim=(0, 3)).cpu().numpy()
            if kernel_sums[stage] is None:
                kernel_sums[stage] = kernel_sum
            else:
                kernel_sums[stage] += kernel_sum
            if effective_sums[stage] is None:
                effective_sums[stage] = effective_sum
            else:
                effective_sums[stage] += effective_sum

            kernel_counts[stage] += kernel.shape[0] * kernel.shape[3]
            effective_counts[stage] += effective.shape[0] * effective.shape[3]

            masks = band_masks(k2)
            for band, mask in masks.items():
                values = effective[:, mask, :].mean(dim=1)
                band_stats[(stage, band)].update(values)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, phy_field):
            stage = stage_from_name(name)
            if stage is not None:
                hooks.append(module.register_forward_pre_hook(make_hook(stage, module)))

    processed = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(args.device, non_blocking=True)
            model(images)
            processed += images.shape[0]
            if processed % max(args.batch_size * 10, 1) == 0:
                print(f"processed {processed} images")

    for hook in hooks:
        hook.remove()

    effective_maps = {
        stage: effective_sums[stage] / max(effective_counts[stage], 1)
        for stage in effective_sums
    }
    kernel_maps = {
        stage: kernel_sums[stage] / max(kernel_counts[stage], 1)
        for stage in kernel_sums
    }

    save_heatmap_grid(
        effective_maps,
        output_dir / "effective_mask_heatmap.png",
        "Stage-wise average effective spectral mask",
        args.dpi,
    )
    save_heatmap_grid(
        kernel_maps,
        output_dir / "kernel_only_heatmap.png",
        "Stage-wise average learned spectral kernel",
        args.dpi,
    )
    save_band_plot(band_stats, output_dir / "frequency_band_response.png", args.dpi)
    save_csv(band_stats, output_dir / "frequency_band_response.csv")

    for stage in sorted(effective_maps):
        np.save(output_dir / f"stage{stage + 1}_effective_mask.npy", effective_maps[stage])
        np.save(output_dir / f"stage{stage + 1}_kernel.npy", kernel_maps[stage])

    print(f"processed images: {processed}")
    print(f"saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
