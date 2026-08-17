#!/usr/bin/env python3
import os
import shutil
from io import BytesIO
from pathlib import Path

os.environ.setdefault("TMPDIR", "/data/sunmy/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/data/sunmy/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path('/data/sunmy/visual/autodriving/BDD100K')
TOP10 = OUT_DIR / 'top10_grids'
REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = REPO_ROOT / 'assets' / 'qualitative'

# Values are BDD100K val bbox AP from the corresponding logs, multiplied by 100.
CLASS_LABELS = ['person', 'rider', 'car', 'truck', 'bus', 'motor.', 'bicycle', 'light', 'sign', 'mAP']
METRICS = {
    'Swin-T':      [32.8, 24.3, 47.3, 44.8, 47.2, 23.0, 24.6, 22.1, 35.9, 30.2],
    'ConvNeXt-T': [34.7, 26.4, 48.7, 45.7, 49.1, 23.8, 25.7, 22.9, 37.2, 31.7],
    'VPF':        [35.2, 27.2, 48.5, 46.2, 47.7, 26.2, 26.0, 23.1, 37.5, 31.8],
}
COLORS = {'Swin-T': '#b9972a', 'ConvNeXt-T': '#f2a033', 'VPF': '#df4f2b'}
SELECTED = [
    TOP10 / '05_b2f4ebb7-ceaa12dc.jpg',
    TOP10 / '01_b1d7b3ac-2a92e19f.jpg',
]
METHODS = ['Input', 'GT', 'Swin-T', 'ConvNeXt-T', 'VPF']


def font(size, bold=False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

FONT_PANEL = font(30, True)
FONT_LABEL = font(26, True)


def fig_to_image(fig, size):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGBA').resize(size, Image.Resampling.LANCZOS)


def make_bar_chart(size):
    labels = CLASS_LABELS
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12.8, 3.2))
    for idx, (name, vals) in enumerate(METRICS.items()):
        ax.bar(x + (idx - 1) * width, vals, width, label=name, color=COLORS[name])
    ax.set_ylabel('AP / mAP (%)', fontsize=12)
    ax.set_ylim(0, 55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=0.6, alpha=0.7)
    fig.patch.set_alpha(0.0)
    return fig_to_image(fig, size)


def crop_grid_panels(path):
    img = Image.open(path).convert('RGB')
    # Grid produced by visualize_bdd100k_selection.py: label_h=38, panel_w=440, panel_h=248, gap=12.
    panel_w, panel_h, gap, label_h = 440, 248, 12, 38
    panels = []
    for i in range(5):
        x = i * (panel_w + gap)
        panels.append(img.crop((x, label_h, x + panel_w, label_h + panel_h)))
    return panels


def make_figure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width = 2050
    chart_h = 540
    gap = 14
    margin = 24
    panel_w, panel_h = 390, 219
    label_h = 38
    row_gap = 16
    bottom_h = label_h + 2 * panel_h + row_gap + 28
    height = chart_h + 34 + bottom_h

    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)

    # Top chart band. Keep the canvas transparent for paper placement.
    draw.text((12, 8), 'a', font=FONT_PANEL, fill=(0, 0, 0, 255))
    chart = make_bar_chart((width - 110, chart_h - 50))
    canvas.paste(chart, (76, 18), chart)

    # Qualitative band, styled after the reference figure but with transparent background.
    y0 = chart_h + 34
    draw.text((12, y0 + 8), 'b', font=FONT_PANEL, fill=(0, 0, 0, 255))

    total_w = 5 * panel_w + 4 * gap
    x0 = (width - total_w) // 2
    for i, name in enumerate(METHODS):
        x = x0 + i * (panel_w + gap)
        bbox = draw.textbbox((0, 0), name, font=FONT_LABEL)
        draw.text((x + (panel_w - (bbox[2] - bbox[0])) // 2, y0 + 8), name, font=FONT_LABEL, fill=(0, 0, 0, 255))

    for row_idx, path in enumerate(SELECTED):
        panels = crop_grid_panels(path)
        row_y = y0 + label_h + row_idx * (panel_h + row_gap)
        for col_idx, panel in enumerate(panels):
            x = x0 + col_idx * (panel_w + gap)
            panel = panel.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
            canvas.paste(panel.convert('RGBA'), (x, row_y))
            if col_idx == 4:
                draw.rectangle((x, row_y, x + panel_w - 1, row_y + panel_h - 1), outline=(255, 128, 0, 255), width=5)

    out = OUT_DIR / 'bdd100k_reference_style_figure.png'
    canvas.save(out)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, PAPER_FIG_DIR / out.name)
    print(out)


if __name__ == '__main__':
    make_figure()
