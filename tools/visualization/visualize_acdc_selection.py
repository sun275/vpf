#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("TMPDIR", "/data/sunmy/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/data/sunmy/tmp/matplotlib")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vpf.adapters.mmsegmentation import MMSEG_VPF  # noqa: F401,E402
from mmseg.apis import inference_model, init_model  # noqa: E402

CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)

PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

DEFAULT_MODELS = {
    "vpf": (
        "configs/vpf/upernet_vpf_40k_acdc_512x1024_tiny.py",
        "/data/sunmy/vpf_seg_acdc/vpf_tiny_2342_40k/iter_40000.pth",
    ),
    "swin": (
        "configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-40k_acdc-512x1024.py",
        "/data/sunmy/vpf_seg_acdc/swin_tiny_40k/iter_40000.pth",
    ),
    "convnext": (
        "configs/convnext/convnext-tiny_upernet_8xb2-40k_acdc-512x1024.py",
        "/data/sunmy/vpf_seg_acdc/convnext_tiny_40k/iter_40000.pth",
    ),
    "r50": (
        "configs/upernet/upernet_r50_8xb2-40k_acdc-512x1024.py",
        "/data/sunmy/vpf_seg_acdc/r50_40k/iter_40000.pth",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Select ACDC examples where VPF beats baselines.")
    parser.add_argument("--data-root", default="/data1/sunmy/ACDC/mmseg_format")
    parser.add_argument("--output-dir", default="/data/sunmy/visual/autodriving/ACDC")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--min-vpf-miou", type=float, default=0.0)
    parser.add_argument("--min-valid-classes", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def list_acdc_val(data_root, max_images=0):
    image_root = Path(data_root) / "leftImg8bit" / "val"
    gt_root = Path(data_root) / "gtFine" / "val"
    images = sorted(image_root.rglob("*_leftImg8bit.png"))
    if max_images > 0:
        images = images[:max_images]
    samples = []
    for image_path in images:
        rel = image_path.relative_to(image_root)
        gt_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        gt_path = gt_root / rel.parent / gt_name
        if gt_path.is_file():
            samples.append((image_path, gt_path, rel))
    return samples


def colorize_mask(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    color[valid] = PALETTE[mask[valid]]
    return color


def per_image_iou(pred, gt, num_classes=19, ignore_index=255):
    valid = gt != ignore_index
    pred = pred[valid]
    gt = gt[valid]
    ious = []
    valid_classes = []
    for cls in range(num_classes):
        pred_c = pred == cls
        gt_c = gt == cls
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_c, gt_c).sum()
        ious.append(inter / union)
        valid_classes.append(cls)
    if not ious:
        return 0.0, [], {}
    cls_iou = {CLASSES[c]: iou * 100.0 for c, iou in zip(valid_classes, ious)}
    return float(np.mean(ious) * 100.0), valid_classes, cls_iou


def cache_stem(rel):
    return rel.as_posix().replace("/", "__").replace("_leftImg8bit.png", "")


def prediction_cache_path(output_dir, model_name, rel):
    return Path(output_dir) / "pred_cache" / model_name / f"{cache_stem(rel)}.png"


def run_predictions(args, samples):
    output_dir = Path(args.output_dir)
    predictions = {name: {} for name in DEFAULT_MODELS}

    for model_name, (config, checkpoint) in DEFAULT_MODELS.items():
        config_path = ROOT / config
        checkpoint_path = Path(checkpoint)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        missing = [
            (image_path, rel)
            for image_path, _, rel in samples
            if args.force or not prediction_cache_path(output_dir, model_name, rel).is_file()
        ]

        if missing:
            print(f"[{model_name}] loading model: {checkpoint_path}", flush=True)
            seg_model = init_model(str(config_path), str(checkpoint_path), device=args.device)
            for idx, (image_path, rel) in enumerate(missing, 1):
                result = inference_model(seg_model, str(image_path))
                pred = result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                cache_path = prediction_cache_path(output_dir, model_name, rel)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred).save(cache_path)
                if idx % 25 == 0 or idx == len(missing):
                    print(f"[{model_name}] predicted {idx}/{len(missing)}", flush=True)
            del seg_model
        else:
            print(f"[{model_name}] using cached predictions", flush=True)

        for _, _, rel in samples:
            predictions[model_name][rel.as_posix()] = np.array(Image.open(prediction_cache_path(output_dir, model_name, rel)))

    return predictions


def rank_samples(samples, predictions, output_dir):
    rows = []
    for image_path, gt_path, rel in samples:
        gt = np.array(Image.open(gt_path))
        gt_classes = sorted(int(c) for c in np.unique(gt) if c != 255 and 0 <= int(c) < 19)
        metrics = {}
        for model_name in DEFAULT_MODELS:
            miou, _, _ = per_image_iou(predictions[model_name][rel.as_posix()], gt)
            metrics[model_name] = miou
        baseline_scores = [metrics["swin"], metrics["convnext"], metrics["r50"]]
        weather = rel.parts[0] if rel.parts else "unknown"
        row = {
            "rel": rel.as_posix(),
            "weather": weather,
            "image_path": str(image_path),
            "gt_path": str(gt_path),
            "valid_classes": len(gt_classes),
            "vpf_miou": metrics["vpf"],
            "swin_miou": metrics["swin"],
            "convnext_miou": metrics["convnext"],
            "r50_miou": metrics["r50"],
            "best_baseline_miou": max(baseline_scores),
            "delta_best": metrics["vpf"] - max(baseline_scores),
            "delta_mean": metrics["vpf"] - float(np.mean(baseline_scores)),
            "gt_classes": " ".join(CLASSES[c] for c in gt_classes),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["delta_best"], reverse=True)
    csv_path = Path(output_dir) / "acdc_per_image_miou.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved ranking csv: {csv_path}")
    return rows


def draw_comparison(row, predictions, output_dir, dpi=180):
    image_path = Path(row["image_path"])
    gt_path = Path(row["gt_path"])
    rel = Path(row["rel"])
    image = np.array(Image.open(image_path).convert("RGB"))
    gt = np.array(Image.open(gt_path))

    panels = [
        ("Input", image),
        ("GT", colorize_mask(gt)),
        (f"VPF {row['vpf_miou']:.1f}", colorize_mask(predictions["vpf"][rel.as_posix()])),
        (f"Swin {row['swin_miou']:.1f}", colorize_mask(predictions["swin"][rel.as_posix()])),
        (f"ConvNeXt {row['convnext_miou']:.1f}", colorize_mask(predictions["convnext"][rel.as_posix()])),
        (f"R50 {row['r50_miou']:.1f}", colorize_mask(predictions["r50"][rel.as_posix()])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8))
    for ax, (title, panel) in zip(axes.ravel(), panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=13)
        ax.set_axis_off()
    fig.suptitle(
        f"{rel.as_posix()} | delta_best={row['delta_best']:.1f}, valid_classes={row['valid_classes']}",
        fontsize=12,
        y=0.985,
    )
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.96))
    out_name = cache_stem(rel)
    out_path = Path(output_dir) / "top30_grids" / f"{out_name}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    samples = list_acdc_val(args.data_root, args.max_images)
    if not samples:
        raise RuntimeError(f"No ACDC val samples found under {args.data_root}")
    print(f"found {len(samples)} ACDC val samples", flush=True)

    predictions = run_predictions(args, samples)
    rows = rank_samples(samples, predictions, args.output_dir)

    candidates = [
        row for row in rows
        if row["vpf_miou"] >= args.min_vpf_miou
        and int(row["valid_classes"]) >= args.min_valid_classes
    ][:args.topk]
    selected_csv = Path(args.output_dir) / "selected_top30.csv"
    with selected_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"saved selected csv: {selected_csv}")

    for idx, row in enumerate(candidates, 1):
        out_path = draw_comparison(row, predictions, args.output_dir, args.dpi)
        print(f"rendered {idx}/{len(candidates)}: {out_path}", flush=True)


if __name__ == "__main__":
    main()
