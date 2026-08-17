#!/usr/bin/env python3
import os
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault("TMPDIR", "/data/sunmy/tmp")


DATA_ROOT = Path("/data1/sunmy/cityscapes")
VIS_ROOT = Path("/data/sunmy/visual/autodriving/cityscapes")
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

MODELS = [
    ("GT", None),
    ("VPF", "vpf"),
    ("ConvNeXt-T", "convnext"),
    ("Swin-T", "swin"),
]

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "light", "sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

METRICS = {
    "VPF": {"mIoU": 80.92, "mAcc": 86.90, "aAcc": 96.57},
    "ConvNeXt-T": {"mIoU": 80.33, "mAcc": 86.72, "aAcc": 96.53},
    "ResNet-50": {"mIoU": 79.19, "mAcc": 85.66, "aAcc": 96.21},
    "Swin-T": {"mIoU": 79.04, "mAcc": 85.78, "aAcc": 96.37},
}

CLASS_IOU = {
    "VPF": [
        98.37, 87.01, 93.22, 58.59, 63.60, 69.50, 75.72, 82.67, 93.19,
        66.95, 95.52, 83.79, 62.14, 95.95, 84.08, 91.86, 82.61, 72.60, 80.10,
    ],
    "ConvNeXt-T": [
        98.37, 86.69, 93.27, 56.63, 64.20, 70.02, 75.04, 83.00, 93.08,
        66.27, 95.37, 84.12, 63.93, 95.76, 79.69, 90.36, 79.51, 70.60, 80.39,
    ],
    "ResNet-50": [
        98.09, 84.68, 92.75, 46.34, 60.36, 68.15, 73.84, 82.34, 92.62,
        63.50, 95.05, 83.49, 62.34, 95.73, 79.60, 91.13, 84.61, 70.21, 79.80,
    ],
    "Swin-T": [
        98.18, 85.42, 93.09, 56.79, 63.19, 68.18, 74.64, 82.07, 93.04,
        64.87, 95.29, 83.73, 62.13, 95.84, 76.51, 86.71, 73.50, 68.97, 79.62,
    ],
}

# ROI coordinates are on the original 2048x1024 Cityscapes frame.
CASES = [
    {
        "city": "frankfurt",
        "stem": "frankfurt_000000_014480",
        "title": "a",
        "roi": (120, 300, 700, 590),
    },
    {
        "city": "munster",
        "stem": "munster_000147_000019",
        "title": "b",
        "roi": (720, 300, 1370, 625),
    },
]

CASE = CASES[0]
MODEL_COLORS = {
    "VPF": "#df4f2b",
    "ConvNeXt-T": "#f2a033",
    "ResNet-50": "#8b9292",
    "Swin-T": "#b9972a",
    "Best baseline": "#d899b1",
}


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(28, bold=True)
FONT_PANEL = font(24, bold=True)
FONT_SMALL = font(22, bold=True)


def image_path(case):
    return DATA_ROOT / "leftImg8bit" / "val" / case["city"] / f"{case['stem']}_leftImg8bit.png"


def gt_path(case):
    return DATA_ROOT / "gtFine" / "val" / case["city"] / f"{case['stem']}_gtFine_labelTrainIds.png"


def pred_path(case, model_name):
    return PRED_ROOT / model_name / f"{case['city']}__{case['stem']}.png"


def colorize(mask):
    mask = np.asarray(mask)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    out[valid] = PALETTE[mask[valid]]
    return Image.fromarray(out, mode="RGB")


def expand_roi_to_aspect(roi, image_size, aspect=2.0):
    x1, y1, x2, y2 = roi
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    img_w, img_h = image_size
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > img_w:
        x1 -= x2 - img_w
        x2 = img_w
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > img_h:
        y1 -= y2 - img_h
        y2 = img_h
    return tuple(int(round(v)) for v in (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))


def resize_fill(img, size):
    return img.resize(size, Image.Resampling.NEAREST if img.mode != "RGB" else Image.Resampling.BICUBIC)


def draw_label(draw, xy, text, fill="black", fnt=FONT_PANEL):
    draw.text(xy, text, font=fnt, fill=fill)


def load_panels(case):
    original = Image.open(image_path(case)).convert("RGB")
    gt = colorize(Image.open(gt_path(case)))
    panels = [("Input", original), ("GT", gt)]
    for label, model_name in MODELS[1:]:
        panels.append((label, colorize(Image.open(pred_path(case, model_name)))))
    return panels


def load_paper_panels(case):
    original = Image.open(image_path(case)).convert("RGB")
    return [
        ("Input", original),
        ("VPF", colorize(Image.open(pred_path(case, "vpf")))),
        ("ConvNeXt-T", colorize(Image.open(pred_path(case, "convnext")))),
        ("Swin-T", colorize(Image.open(pred_path(case, "swin")))),
    ]


def draw_case(case):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = load_panels(case)
    original_size = panels[0][1].size
    roi = expand_roi_to_aspect(case["roi"], original_size)

    panel_w, panel_h = 430, 215
    zoom_w, zoom_h = 430, 215
    gap = 12
    margin_x = 20
    margin_top = 42
    title_h = 38
    row_gap = 16
    bottom = 20
    width = margin_x * 2 + panel_w * len(panels) + gap * (len(panels) - 1)
    height = margin_top + title_h + panel_h + row_gap + zoom_h + bottom
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (margin_x, 4), case["title"], fnt=FONT_TITLE)

    for idx, (label, panel) in enumerate(panels):
        x = margin_x + idx * (panel_w + gap)
        y = margin_top
        full = resize_fill(panel, (panel_w, panel_h))
        canvas.paste(full.convert("RGBA"), (x, y + title_h))
        draw_label(draw, (x + 8, y + 3), label, fnt=FONT_SMALL)

        scale_x = panel_w / original_size[0]
        scale_y = panel_h / original_size[1]
        rx1, ry1, rx2, ry2 = roi
        box = (
            x + int(rx1 * scale_x),
            y + title_h + int(ry1 * scale_y),
            x + int(rx2 * scale_x),
            y + title_h + int(ry2 * scale_y),
        )
        draw.rectangle(box, outline=(255, 128, 0), width=5)

        crop = panel.crop(roi)
        zoom = resize_fill(crop, (zoom_w, zoom_h))
        zy = y + title_h + panel_h + row_gap
        canvas.paste(zoom.convert("RGBA"), (x, zy))
        draw.rectangle((x, zy, x + zoom_w - 1, zy + zoom_h - 1), outline=(255, 128, 0), width=6)

    out_path = OUT_DIR / f"{case['title']}_{case['city']}__{case['stem']}.png"
    canvas.save(out_path)
    return out_path


def read_per_image_miou():
    csv_path = VIS_ROOT / "cityscapes_per_image_miou.csv"
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "VPF": float(row["vpf_miou"]),
                "ConvNeXt-T": float(row["convnext_miou"]),
                "Swin-T": float(row["swin_miou"]),
                "ResNet-50": float(row["r50_miou"]),
            })
    return rows


def draw_boxplot_pair(baseline):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_per_image_miou()
    if baseline == "Best baseline":
        baseline_data = [max(r["ConvNeXt-T"], r["Swin-T"], r["ResNet-50"]) for r in rows]
        baseline_label = "Best-B"
    else:
        baseline_data = [r[baseline] for r in rows]
        baseline_label = baseline.replace("-T", "")
    data = [baseline_data, [r["VPF"] for r in rows]]

    fig, ax = plt.subplots(figsize=(2.0, 2.65), dpi=300)
    bp = ax.boxplot(
        data,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#ffff33", linewidth=2.0),
        whiskerprops=dict(color="black", linewidth=1.6),
        capprops=dict(color="black", linewidth=1.6),
        boxprops=dict(color="black", linewidth=1.6),
    )
    for patch, color in zip(bp["boxes"], [MODEL_COLORS[baseline], MODEL_COLORS["VPF"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([baseline_label, "VPF"], fontsize=9)
    ax.set_ylabel("IoU (%)", fontsize=9)
    ax.set_ylim(40, 95)
    ax.set_yticks([40, 55, 70, 85, 95])
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax.tick_params(axis="both", width=1.2, labelsize=9)
    fig.tight_layout(pad=0.28)
    out = OUT_DIR / f"cityscapes_box_{baseline.lower().replace('-', '').replace(' ', '_')}_vs_vpf.png"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out


def draw_class_heatmap():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Swin-T", "ConvNeXt-T", "ResNet-50", "VPF"]
    data = np.array([CLASS_IOU[label] for label in labels])

    fig, ax = plt.subplots(figsize=(8.3, 2.4), dpi=300)
    im = ax.imshow(data, cmap="Oranges", vmin=45, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] < 62 else "black"
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6.3, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("")
    fig.tight_layout(pad=0.35)
    out = OUT_DIR / "cityscapes_per_class_iou_heatmap.png"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out


def draw_delta_to_best_baseline():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compare = ["Swin-T", "ConvNeXt-T", "VPF"]
    names = CLASS_NAMES + ["mIoU"]
    y = np.arange(len(names))
    height = 0.18

    fig, ax = plt.subplots(figsize=(3.35, 4.65), dpi=300)
    for offset, name in zip([-height, 0, height], compare):
        vals = CLASS_IOU[name] + [METRICS[name]["mIoU"]]
        ax.barh(y + offset, vals, height=height, label=name, color=MODEL_COLORS[name])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("IoU / mIoU (%)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    ax.tick_params(axis="both", width=1.1)
    fig.tight_layout(pad=0.35)
    out = OUT_DIR / "cityscapes_delta_to_best_baseline.png"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out


def draw_full_case(case):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = load_paper_panels(case)
    panel_w, panel_h = 430, 215
    gap = 12
    margin_x = 20
    margin_top = 34
    title_h = 34
    bottom = 18
    width = margin_x * 2 + panel_w * len(panels) + gap * (len(panels) - 1)
    height = margin_top + title_h + panel_h + bottom
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (margin_x, 2), case["title"], fnt=FONT_TITLE)

    for idx, (label, panel) in enumerate(panels):
        x = margin_x + idx * (panel_w + gap)
        y = margin_top
        full = resize_fill(panel, (panel_w, panel_h))
        canvas.paste(full.convert("RGBA"), (x, y + title_h))
        draw_label(draw, (x + 8, y), label, fnt=FONT_SMALL)

    out_path = OUT_DIR / f"{case['title']}_{case['city']}__{case['stem']}_full.png"
    canvas.save(out_path)
    return out_path


def concat_vertical(paths):
    images = [Image.open(p).convert("RGBA") for p in paths]
    gap = 18
    width = max(im.width for im in images)
    height = sum(im.height for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    y = 0
    for im in images:
        canvas.paste(im, ((width - im.width) // 2, y))
        y += im.height + gap
    out_path = OUT_DIR / "cityscapes_qualitative_four_cases.png"
    canvas.save(out_path)
    return out_path


def concat_horizontal(paths, out_name, bg="white", gap=22, pad=18):
    images = [Image.open(p).convert("RGB") for p in paths]
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + gap * (len(images) - 1) + pad * 2
    canvas = Image.new("RGB", (width, height + pad * 2), bg)
    x = pad
    for im in images:
        canvas.paste(im, (x, pad + (height - im.height) // 2))
        x += im.width + gap
    out_path = OUT_DIR / out_name
    canvas.save(out_path)
    return out_path


def make_paper_style_figure(box_paths, heatmap_path, class_bar_path, case_paths):
    box_imgs = [Image.open(p).convert("RGBA") for p in box_paths]
    heat = Image.open(heatmap_path).convert("RGBA")
    class_bar = Image.open(class_bar_path).convert("RGBA")
    cases = [Image.open(p).convert("RGBA") for p in case_paths]

    target_w = 2100
    margin = 28
    gap = 18
    left_w = 1420
    bar_w = target_w - 2 * margin - gap - left_w
    top_h = 390
    heat_h = 460
    box_w = (left_w - 2 * gap) // 3
    case_h = 310

    def fit_width(im, width):
        h = int(round(im.height * width / im.width))
        return im.resize((width, h), Image.Resampling.LANCZOS)

    def fit_exact(im, size):
        return im.resize(size, Image.Resampling.LANCZOS)

    boxes = []
    for im in box_imgs:
        scale = min(box_w / im.width, top_h / im.height)
        boxes.append(im.resize((int(round(im.width * scale)), int(round(im.height * scale))), Image.Resampling.LANCZOS))
    class_bar = fit_exact(class_bar, (bar_w, top_h + gap + heat_h))
    heat = fit_exact(heat, (left_w, heat_h))
    case_imgs = [fit_width(im, target_w - 2 * margin) for im in cases]
    case_imgs = [
        fit_exact(im, (target_w - 2 * margin, case_h)) if im.height > case_h else im
        for im in case_imgs
    ]

    cases_h = sum(im.height for im in case_imgs) + gap * (len(case_imgs) - 1)
    height = margin + top_h + gap + heat_h + gap + cases_h + margin
    canvas = Image.new("RGBA", (target_w, height), (255, 255, 255, 0))

    x = margin
    for im in boxes:
        canvas.alpha_composite(im, (x + (box_w - im.width) // 2, margin + (top_h - im.height) // 2))
        x += box_w + gap
    canvas.alpha_composite(class_bar, (margin + left_w + gap, margin))

    y = margin + top_h + gap
    canvas.alpha_composite(heat, (margin, y))
    y += heat_h + gap
    for im in case_imgs:
        canvas.alpha_composite(im, (margin, y))
        y += im.height + gap

    out_path = OUT_DIR / "cityscapes_reference_style_figure.png"
    canvas.save(out_path)
    return out_path


def main():
    box_paths = [
        draw_boxplot_pair("ResNet-50"),
        draw_boxplot_pair("Swin-T"),
        draw_boxplot_pair("ConvNeXt-T"),
    ]
    heatmap_path = draw_class_heatmap()
    class_bar_path = draw_delta_to_best_baseline()
    small_paths = [
        draw_full_case(CASES[0]),
        draw_full_case(CASES[1]),
    ]
    big_path = concat_vertical(small_paths)
    full_path = make_paper_style_figure(box_paths, heatmap_path, class_bar_path, small_paths)
    print("small figures:")
    for p in box_paths:
        print(p)
    print(heatmap_path)
    print(class_bar_path)
    for p in small_paths:
        print(p)
    print("large figure:")
    print(big_path)
    print(full_path)


if __name__ == "__main__":
    main()
