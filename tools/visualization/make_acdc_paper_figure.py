#!/usr/bin/env python3
import csv
import os
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault("TMPDIR", "/data/sunmy/tmp")

DATA_ROOT = Path("/data1/sunmy/ACDC/mmseg_format")
VIS_ROOT = Path("/data/sunmy/visual/autodriving/ACDC")
PRED_ROOT = VIS_ROOT / "pred_cache"
OUT_DIR = VIS_ROOT / "paper"

PALETTE = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
], dtype=np.uint8)

CASES = [
    ("Rain", "rain/GP010400/GP010400_frame_000350_leftImg8bit.png"),
    ("Night", "night/GOPR0351/GOPR0351_frame_000809_leftImg8bit.png"),
]

MODELS = [
    ("Input", None),
    ("GT", "gt"),
    ("Swin-T", "swin"),
    ("ConvNeXt-T", "convnext"),
    ("ResNet-50", "r50"),
    ("VPF", "vpf"),
]

METHODS = [
    ("ResNet-50", "r50", "#8b9292"),
    ("Swin-T", "swin", "#b9972a"),
    ("ConvNeXt-T", "convnext", "#f2a033"),
    ("VPF", "vpf", "#df4f2b"),
]

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "light", "sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(32, True)
FONT_LABEL = font(24, True)
FONT_SMALL = font(22, True)


def pred_key(rel):
    path = Path(rel)
    stem = path.name.replace("_leftImg8bit.png", "")
    return f"{path.parts[0]}__{path.parts[1]}__{stem}.png"


def gt_path(rel):
    path = Path(rel)
    name = path.name.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
    return DATA_ROOT / "gtFine" / "val" / path.parts[0] / path.parts[1] / name


def colorize(mask):
    arr = np.asarray(mask)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    valid = (arr >= 0) & (arr < len(PALETTE))
    out[valid] = PALETTE[arr[valid]]
    return Image.fromarray(out, "RGB")


def resize_panel(img, size, is_mask=False):
    method = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    return img.resize(size, method)


def read_scores():
    scores = {}
    with (VIS_ROOT / "selected_top30.csv").open() as f:
        for row in csv.DictReader(f):
            scores[row["rel"]] = {
                "vpf": float(row["vpf_miou"]),
                "swin": float(row["swin_miou"]),
                "convnext": float(row["convnext_miou"]),
                "r50": float(row["r50_miou"]),
            }
    return scores


def read_metric_rows():
    rows = []
    with (VIS_ROOT / "acdc_per_image_miou.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "rel": row["rel"],
                "weather": row["weather"],
                "vpf": float(row["vpf_miou"]),
                "swin": float(row["swin_miou"]),
                "convnext": float(row["convnext_miou"]),
                "r50": float(row["r50_miou"]),
            })
    return rows


def load_panel(rel, model_name):
    if model_name is None:
        return Image.open(DATA_ROOT / "leftImg8bit" / "val" / rel).convert("RGB"), False
    if model_name == "gt":
        return colorize(Image.open(gt_path(rel))), True
    return colorize(Image.open(PRED_ROOT / model_name / pred_key(rel))), True


def fig_to_image(fig, size):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA").resize(size, Image.Resampling.LANCZOS)


def make_weather_boxplots(rows, size):
    weathers = ["fog", "rain", "night", "snow"]
    fig, axes = plt.subplots(1, 4, figsize=(9.8, 3.2), sharey=True)
    for ax, weather in zip(axes, weathers):
        subset = [r for r in rows if r["weather"] == weather]
        best = [max(r["swin"], r["convnext"], r["r50"]) for r in subset]
        vpf = [r["vpf"] for r in subset]
        bp = ax.boxplot(
            [best, vpf],
            labels=["Best\nbase", "VPF"],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="#fff23a", linewidth=2.0),
            boxprops=dict(linewidth=1.4),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], ["#d8a2b4", "#df4f2b"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
        ax.set_title(weather.capitalize(), fontsize=11, weight="bold")
        ax.set_ylim(30, 100)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)
    axes[0].set_ylabel("per-image mIoU (%)", fontsize=10)
    fig.patch.set_alpha(0.0)
    return fig_to_image(fig, size)


def make_weather_bars(rows, size):
    weathers = ["fog", "rain", "night", "snow"]
    y = np.arange(len(weathers))
    fig, ax = plt.subplots(figsize=(5.7, 3.2))
    height = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, (label, key, color) in zip(offsets, METHODS):
        vals = [np.mean([r[key] for r in rows if r["weather"] == w]) for w in weathers]
        ax.barh(y + offset * height, vals, height=height, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([w.capitalize() for w in weathers], fontsize=10)
    ax.set_xlim(45, 90)
    ax.set_xlabel("mean mIoU (%)", fontsize=10)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=False, fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0.0)
    return fig_to_image(fig, size)


def compute_class_iou(rows):
    result = {}
    for _, key, _ in METHODS:
        inter = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        union = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for row in rows:
            gt = np.asarray(Image.open(gt_path(row["rel"])))
            pred = np.asarray(Image.open(PRED_ROOT / key / pred_key(row["rel"])))
            valid = gt != 255
            for cls in range(len(CLASS_NAMES)):
                pred_cls = (pred == cls) & valid
                gt_cls = (gt == cls) & valid
                inter[cls] += np.logical_and(pred_cls, gt_cls).sum()
                union[cls] += np.logical_or(pred_cls, gt_cls).sum()
        result[key] = np.divide(inter, union, out=np.full_like(inter, np.nan), where=union > 0) * 100.0
    return result


def make_class_gain_plot(class_iou, size):
    baseline = np.nanmax(np.stack([class_iou["r50"], class_iou["swin"], class_iou["convnext"]]), axis=0)
    gain = class_iou["vpf"] - baseline
    valid = np.isfinite(gain)
    order = np.argsort(gain[valid])[-9:]
    valid_indices = np.where(valid)[0][order]
    names = [CLASS_NAMES[i] for i in valid_indices]
    vals = gain[valid_indices]

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    colors = ["#df4f2b" if v >= 0 else "#8b9292" for v in vals]
    ax.barh(np.arange(len(vals)), vals, color=colors)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("VPF - best baseline IoU (%)", fontsize=10)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0.0)
    return fig_to_image(fig, size)


def make_figure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = read_scores()

    panel_w, panel_h = 390, 219
    gap_x, gap_y = 10, 32
    margin_l, margin_r = 18, 18
    margin_t, margin_b = 54, 18
    label_h = 34
    row_label_w = 88

    width = margin_l + row_label_w + len(MODELS) * panel_w + (len(MODELS) - 1) * gap_x + margin_r
    height = margin_t + len(CASES) * (label_h + panel_h) + (len(CASES) - 1) * gap_y + margin_b
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin_l, 8), "ACDC adverse-condition semantic segmentation", font=FONT_TITLE, fill=(20, 20, 20, 255))

    orange = (255, 128, 0, 255)
    for row_idx, (weather, rel) in enumerate(CASES):
        y0 = margin_t + row_idx * (label_h + panel_h + gap_y)
        draw.text((margin_l, y0 + label_h + panel_h // 2 - 14), weather, font=FONT_LABEL, fill=(25, 25, 25, 255))
        for col_idx, (label, model_name) in enumerate(MODELS):
            x0 = margin_l + row_label_w + col_idx * (panel_w + gap_x)
            score = ""
            if model_name in {"vpf", "swin", "convnext", "r50"}:
                score = f" {scores[rel][model_name]:.1f}"
            draw.text((x0 + 4, y0), f"{label}{score}", font=FONT_SMALL, fill=(25, 25, 25, 255))

            panel, is_mask = load_panel(rel, model_name)
            panel = resize_panel(panel, (panel_w, panel_h), is_mask)
            canvas.paste(panel.convert("RGBA"), (x0, y0 + label_h))
            if model_name == "vpf":
                draw.rectangle(
                    (x0, y0 + label_h, x0 + panel_w - 1, y0 + label_h + panel_h - 1),
                    outline=orange,
                    width=5,
                )

    out_path = OUT_DIR / "acdc_rain_night_qualitative.png"
    canvas.save(out_path)
    print(out_path)


def make_summary_figure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_metric_rows()
    scores = read_scores()
    class_iou = compute_class_iou(rows)

    chart_h = 420
    box_img = make_weather_boxplots(rows, (900, chart_h))
    weather_img = make_weather_bars(rows, (570, chart_h))
    class_img = make_class_gain_plot(class_iou, (570, chart_h))

    panel_w, panel_h = 315, 177
    gap_x, gap_y = 9, 28
    margin_l, margin_r = 18, 18
    margin_t, margin_b = 54, 20
    title_h = 30
    row_label_w = 82
    chart_gap = 18
    top_h = chart_h
    qualitative_y = margin_t + top_h + 36
    qualitative_w = row_label_w + len(MODELS) * panel_w + (len(MODELS) - 1) * gap_x
    width = margin_l + max(900 + chart_gap + 570 + chart_gap + 570, qualitative_w) + margin_r
    height = qualitative_y + len(CASES) * (title_h + panel_h) + (len(CASES) - 1) * gap_y + margin_b

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_l, 8), "ACDC adverse-condition semantic segmentation", font=FONT_TITLE, fill=(20, 20, 20, 255))

    x = margin_l
    canvas.paste(box_img, (x, margin_t), box_img)
    draw.text((x, margin_t - 28), "a", font=FONT_LABEL, fill=(20, 20, 20, 255))
    x += 900 + chart_gap
    canvas.paste(weather_img, (x, margin_t), weather_img)
    draw.text((x, margin_t - 28), "b", font=FONT_LABEL, fill=(20, 20, 20, 255))
    x += 570 + chart_gap
    canvas.paste(class_img, (x, margin_t), class_img)
    draw.text((x, margin_t - 28), "c", font=FONT_LABEL, fill=(20, 20, 20, 255))

    orange = (255, 128, 0, 255)
    start_x = margin_l
    draw.text((start_x, qualitative_y - 28), "d", font=FONT_LABEL, fill=(20, 20, 20, 255))
    for row_idx, (weather, rel) in enumerate(CASES):
        y0 = qualitative_y + row_idx * (title_h + panel_h + gap_y)
        draw.text((start_x, y0 + title_h + panel_h // 2 - 14), weather, font=FONT_LABEL, fill=(25, 25, 25, 255))
        for col_idx, (label, model_name) in enumerate(MODELS):
            x0 = start_x + row_label_w + col_idx * (panel_w + gap_x)
            score = ""
            if model_name in {"vpf", "swin", "convnext", "r50"}:
                score = f" {scores[rel][model_name]:.1f}"
            draw.text((x0 + 3, y0), f"{label}{score}", font=FONT_SMALL, fill=(25, 25, 25, 255))
            panel, is_mask = load_panel(rel, model_name)
            panel = resize_panel(panel, (panel_w, panel_h), is_mask)
            canvas.paste(panel.convert("RGBA"), (x0, y0 + title_h))
            if model_name == "vpf":
                draw.rectangle(
                    (x0, y0 + title_h, x0 + panel_w - 1, y0 + title_h + panel_h - 1),
                    outline=orange,
                    width=5,
                )

    out_path = OUT_DIR / "acdc_summary_figure.png"
    canvas.save(out_path)
    print(out_path)


if __name__ == "__main__":
    make_figure()
    make_summary_figure()
