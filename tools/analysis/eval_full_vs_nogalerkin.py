#!/usr/bin/env python3
import argparse
import csv
import json
import sys
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

from vpf.models import vpf  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_int_tuple(text):
    return tuple(int(v.strip()) for v in text.split(',') if v.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained 2342 VPF checkpoint as full and no_galerkin.')
    parser.add_argument('--data-root', default='/data1/sunmy/Imagenet1k/data/val')
    parser.add_argument(
        '--checkpoint',
        default='/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth')
    parser.add_argument(
        '--output-dir',
        default=str(CLASSIFICATION_ROOT / 'analysis' / 'gate_outputs_test' / 'full_vs_nogalerkin_eval'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--depths', default='2,3,4,2')
    parser.add_argument('--dims', type=int, default=96)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=0, help='0 means all images.')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--num-bins', type=int, default=15, help='Number of ECE bins.')
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model_ema', 'model', 'state_dict'):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint


def strip_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        for prefix in ('module.', 'model.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = value
    return out


def build_model(args, ablation, state_dict):
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
        ablation=ablation,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f'[{ablation}] missing={len(incompatible.missing_keys)}, '
          f'unexpected={len(incompatible.unexpected_keys)}')
    model.to(args.device)
    model.eval()
    return model


def make_loader(args):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, len(dataset)


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    pred = output.topk(maxk, dim=1).indices.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum(0))
    return res


def compute_ece(confidences, correct, num_bins):
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bins = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            acc = 0.0
            conf = 0.0
        else:
            acc = float(correct[mask].mean())
            conf = float(confidences[mask].mean())
            ece += count / len(confidences) * abs(acc - conf)
        bins.append({
            'bin': i,
            'lower': float(lo),
            'upper': float(hi),
            'count': count,
            'accuracy': acc,
            'confidence': conf,
            'gap': abs(acc - conf) if count > 0 else 0.0,
        })
    return ece, bins


def evaluate(model, loader, args, name, num_images):
    total = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    prob_sum = 0.0
    logit_sum = 0.0
    brier_sum = 0.0
    all_confidences = []
    all_correct = []
    use_amp = args.amp and args.device.startswith('cuda')

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            logits = logits.float()
            probs = logits.softmax(dim=1)
            top_prob, top_pred = probs.max(dim=1)
            top_logit = logits.gather(1, top_pred[:, None]).squeeze(1)
            true_prob = probs.gather(1, targets[:, None]).squeeze(1)
            # Multi-class Brier score: sum_c (p_c - y_c)^2
            brier = probs.square().sum(dim=1) - 2.0 * true_prob + 1.0
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            correct = top_pred.eq(targets)

            bs = images.shape[0]
            total += bs
            loss_sum += loss.item() * bs
            top1_sum += acc1.item()
            top5_sum += acc5.item()
            prob_sum += top_prob.sum().item()
            logit_sum += top_logit.sum().item()
            brier_sum += brier.sum().item()
            all_confidences.append(top_prob.detach().cpu())
            all_correct.append(correct.detach().cpu())

            if batch_idx % 20 == 0:
                print(f'[{name}] processed {total}/{num_images}')

    confidences = torch.cat(all_confidences).numpy()
    correct = torch.cat(all_correct).numpy().astype(bool)
    ece, bins = compute_ece(confidences, correct, args.num_bins)
    return {
        'model': name,
        'num_images': total,
        'nll': loss_sum / total,
        'top1_acc': top1_sum / total * 100.0,
        'top5_acc': top5_sum / total * 100.0,
        'ece': ece,
        'brier': brier_sum / total,
        'mean_top1_prob': prob_sum / total,
        'mean_top1_logit': logit_sum / total,
        'reliability_bins': bins,
    }


def save_results(rows, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'full_vs_nogalerkin_eval.csv'
    csv_rows = [
        {k: v for k, v in row.items() if k != 'reliability_bins'}
        for row in rows
    ]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    cal_path = output_dir / 'calibration_table.csv'
    cal_fields = ['model', 'top1_acc', 'nll', 'ece', 'brier', 'mean_top1_prob',
                  'top5_acc', 'mean_top1_logit']
    with cal_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cal_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in cal_fields})

    bins_path = output_dir / 'reliability_bins.csv'
    with bins_path.open('w', newline='') as f:
        fieldnames = ['model', 'bin', 'lower', 'upper', 'count', 'accuracy',
                      'confidence', 'gap']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for item in row['reliability_bins']:
                item = dict(item)
                item['model'] = row['model']
                writer.writerow(item)

    payload = {
        'data_root': args.data_root,
        'checkpoint': args.checkpoint,
        'rows': rows,
    }
    (output_dir / 'full_vs_nogalerkin_eval.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    save_reliability_diagram(rows, output_dir, args.num_bins)
    print(f'saved to {output_dir}')


def save_reliability_diagram(rows, output_dir, num_bins):
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.2), dpi=180)
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        bins = row['reliability_bins']
        centers = np.array([(b['lower'] + b['upper']) / 2.0 for b in bins])
        width = 1.0 / num_bins * 0.9
        acc = np.array([b['accuracy'] for b in bins])
        conf = np.array([b['confidence'] for b in bins])
        counts = np.array([b['count'] for b in bins])

        ax.bar(centers, acc, width=width, color='#4c78a8', label='accuracy')
        ax.plot([0, 1], [0, 1], '--', color='black', linewidth=1.0, label='ideal')
        non_empty = counts > 0
        ax.scatter(centers[non_empty], conf[non_empty], color='#f58518',
                   s=18, label='confidence')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('confidence')
        ax.set_ylabel('accuracy')
        ax.set_title(
            f'{row["model"]}\\n'
            f'Top-1={row["top1_acc"]:.2f}, ECE={row["ece"]:.3f}, '
            f'NLL={row["nll"]:.3f}')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'reliability_diagram.png', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    loader, num_images = make_loader(args)
    state_dict = strip_prefix(load_checkpoint(args.checkpoint))

    rows = []
    for ablation in ('full', 'no_galerkin'):
        model = build_model(args, ablation, state_dict)
        rows.append(evaluate(model, loader, args, ablation, num_images))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results(rows, args)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
