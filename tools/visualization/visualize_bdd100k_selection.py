#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TMPDIR", "/data/sunmy/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/data/sunmy/tmp/matplotlib")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# Register custom VPF backbone.
from vpf.adapters.mmdetection import MMDET_VPF  # noqa: F401
from mmdet.apis import inference_detector, init_detector


REPO_ROOT = Path(__file__).resolve().parents[2]

CLASSES = (
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle', 'traffic light', 'traffic sign'
)
COLORS = [
    (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32), (250, 170, 30), (220, 220, 0),
]
MODELS = {
    'swin': {
        'label': 'Swin-T',
        'config': str(REPO_ROOT / 'tasks/autonomous_driving/configs/detection/bdd100k/swin_tiny_1x.py'),
        'checkpoint': '/data/sunmy/vpf_det_autodriving/swin_t_bdd100k_faster_rcnn_1x/epoch_12.pth',
    },
    'convnext': {
        'label': 'ConvNeXt-T',
        'config': str(REPO_ROOT / 'tasks/autonomous_driving/configs/detection/bdd100k/convnext_tiny_1x.py'),
        'checkpoint': '/data/sunmy/vpf_det_autodriving/convnext_t_bdd100k_faster_rcnn_1x/epoch_12.pth',
    },
    'vpf': {
        'label': 'VPF',
        'config': str(REPO_ROOT / 'tasks/autonomous_driving/configs/detection/bdd100k/vpf_tiny_3x.py'),
        'checkpoint': '/data/sunmy/vpf_det_autodriving/vpf_tiny_2342_bdd100k_faster_rcnn_ms_3x/epoch_36.pth',
    },
}


def font(size, bold=False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(28, True)
FONT_SMALL = font(18, True)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def get_predictions(result, score_thr):
    inst = result.pred_instances
    bboxes = inst.bboxes.detach().cpu().numpy()
    scores = inst.scores.detach().cpu().numpy()
    labels = inst.labels.detach().cpu().numpy()
    keep = scores >= score_thr
    return bboxes[keep], scores[keep], labels[keep]


def detection_quality(pred, gt_boxes, gt_labels, iou_thr=0.5):
    bboxes, scores, labels = pred
    if len(gt_boxes) == 0:
        return 0.0
    matched = set()
    matched_scores = []
    order = np.argsort(-scores)
    for idx in order:
        same = np.where(gt_labels == labels[idx])[0]
        same = [int(j) for j in same if int(j) not in matched]
        if not same:
            continue
        vals = iou(bboxes[idx], gt_boxes[same])
        best_pos = int(np.argmax(vals))
        if vals[best_pos] >= iou_thr:
            gt_idx = same[best_pos]
            matched.add(gt_idx)
            matched_scores.append(float(scores[idx]))
    recall = len(matched) / max(len(gt_boxes), 1)
    score_mean = float(np.mean(matched_scores)) if matched_scores else 0.0
    # Small precision term prevents selecting images with many noisy boxes.
    precision = len(matched) / max(len(bboxes), 1)
    return 0.70 * recall + 0.20 * score_mean + 0.10 * precision


def draw_detections(img, pred, title, score_thr=0.35, max_boxes=40, highlight=False):
    out = img.copy().convert('RGB')
    draw = ImageDraw.Draw(out)
    w, h = out.size
    bboxes, scores, labels = pred
    order = np.argsort(-scores)[:max_boxes]
    for idx in order:
        box = bboxes[idx]
        label = int(labels[idx])
        score = float(scores[idx])
        color = COLORS[label % len(COLORS)]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f'{CLASSES[label]} {score:.2f}'
        tb = draw.textbbox((0, 0), text, font=FONT_SMALL)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle((x1, ty, x1 + tw + 5, ty + th + 4), fill=color)
        draw.text((x1 + 2, ty + 1), text, fill=(255, 255, 255), font=FONT_SMALL)
    if highlight:
        draw.rectangle((2, 2, w - 3, h - 3), outline=(255, 128, 0), width=7)
    return out


def draw_gt(img, gt_boxes, gt_labels, max_boxes=80):
    out = img.copy().convert('RGB')
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for box, label in list(zip(gt_boxes, gt_labels))[:max_boxes]:
        label = int(label)
        color = COLORS[label % len(COLORS)]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = CLASSES[label]
        tb = draw.textbbox((0, 0), text, font=FONT_SMALL)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle((x1, ty, x1 + tw + 5, ty + th + 4), fill=color)
        draw.text((x1 + 2, ty + 1), text, fill=(255, 255, 255), font=FONT_SMALL)
    return out


def make_grid(image_path, preds, scores, gt_boxes, gt_labels, output_path):
    img = Image.open(image_path).convert('RGB')
    panel_w, panel_h = 440, 248
    label_h = 38
    gap = 12
    panels = [('Input', None), ('GT', 'gt')] + [(MODELS[k]['label'], k) for k in ('swin', 'convnext', 'vpf')]
    width = len(panels) * panel_w + (len(panels) - 1) * gap
    height = label_h + panel_h
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (label, key) in enumerate(panels):
        x = i * (panel_w + gap)
        if key is None:
            panel = img.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
            title = 'Input'
        elif key == 'gt':
            det = draw_gt(img, gt_boxes, gt_labels)
            panel = det.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
            title = 'GT'
        else:
            det = draw_detections(img, preds[key], label, highlight=(key == 'vpf'))
            panel = det.resize((panel_w, panel_h), Image.Resampling.BICUBIC)
            title = label
        tb = draw.textbbox((0, 0), title, font=FONT_TITLE)
        draw.text((x + (panel_w - (tb[2]-tb[0])) // 2, 2), title, fill=(20, 20, 20), font=FONT_TITLE)
        canvas.paste(panel, (x, label_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/data1/sunmy/bdd100k')
    parser.add_argument('--ann-file', default='/data1/sunmy/bdd100k/bdd100k_det_10cls_coco/annotations/val_10cls_coco.json')
    parser.add_argument('--output-dir', default='/data/sunmy/visual/autodriving/BDD100K')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=800)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--score-thr', type=float, default=0.35)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = json.load(open(args.ann_file))
    anns_by_img = {}
    for a in ann['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)
    images = [im for im in ann['images'] if len(anns_by_img.get(im['id'], [])) >= 4]
    images = images[:args.max_samples]
    print(f'candidate images: {len(images)}')

    models = {}
    for key in ('swin', 'convnext', 'vpf'):
        info = MODELS[key]
        print(f'loading {key}: {info["checkpoint"]}')
        models[key] = init_detector(info['config'], info['checkpoint'], device=args.device)

    rows = []
    pred_cache = {}
    for idx, im in enumerate(images, 1):
        img_path = Path(args.data_root) / im['file_name']
        gt_anns = anns_by_img.get(im['id'], [])
        gt_boxes = np.array([xywh_to_xyxy(a['bbox']) for a in gt_anns], dtype=np.float32)
        gt_labels = np.array([a['category_id'] - 1 for a in gt_anns], dtype=np.int64)
        preds = {}
        qs = {}
        for key, model_obj in models.items():
            result = inference_detector(model_obj, str(img_path))
            preds[key] = get_predictions(result, args.score_thr)
            qs[key] = detection_quality(preds[key], gt_boxes, gt_labels)
        best_base = max(qs['swin'], qs['convnext'])
        delta = qs['vpf'] - best_base
        rows.append({
            'file_name': im['file_name'],
            'image_id': im['id'],
            'vpf_q': qs['vpf'],
            'swin_q': qs['swin'],
            'convnext_q': qs['convnext'],
            'delta_best': delta,
            'num_gt': len(gt_boxes),
        })
        pred_cache[im['file_name']] = (img_path, preds, qs, gt_boxes, gt_labels)
        if idx % 50 == 0 or idx == len(images):
            print(f'processed {idx}/{len(images)}')

    rows.sort(key=lambda r: (r['delta_best'], r['vpf_q'], r['num_gt']), reverse=True)
    csv_path = out_dir / 'bdd100k_detection_selection.csv'
    with csv_path.open('w') as f:
        f.write('rank,file_name,image_id,num_gt,vpf_q,swin_q,convnext_q,delta_best\n')
        for rank, r in enumerate(rows, 1):
            f.write(f"{rank},{r['file_name']},{r['image_id']},{r['num_gt']},{r['vpf_q']:.6f},{r['swin_q']:.6f},{r['convnext_q']:.6f},{r['delta_best']:.6f}\n")
    print(f'saved ranking: {csv_path}')

    grid_dir = out_dir / 'top10_grids'
    for rank, r in enumerate(rows[:args.topk], 1):
        img_path, preds, qs, gt_boxes, gt_labels = pred_cache[r['file_name']]
        stem = Path(r['file_name']).stem
        out_path = grid_dir / f'{rank:02d}_{stem}.jpg'
        make_grid(img_path, preds, qs, gt_boxes, gt_labels, out_path)
        print(f'rendered {rank}/{args.topk}: {out_path}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
