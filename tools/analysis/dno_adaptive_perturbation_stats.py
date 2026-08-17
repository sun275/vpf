#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TVF


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
        description='Small-batch perturbation statistics for DNO adaptive spectral response.')
    parser.add_argument(
        '--data-root',
        default='/data1/sunmy/Imagenet1k/data/val',
        help='ImageFolder root, e.g. ImageNet validation root.')
    parser.add_argument(
        '--checkpoint',
        default='/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth')
    parser.add_argument(
        '--output-dir',
        default=str(CLASSIFICATION_ROOT / 'analysis' / 'dno_adaptive_stats'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=512)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--depths', default='2,3,4,2')
    parser.add_argument('--dims', type=int, default=96)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--blur-kernel', type=int, default=11)
    parser.add_argument('--blur-sigma', type=float, default=2.0)
    parser.add_argument('--noise-std', type=float, default=0.08)
    parser.add_argument('--edge-strength', type=float, default=1.5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=180)
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
        ablation='full',
    )
    state_dict = strip_prefix(load_checkpoint(args.checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f'loaded checkpoint: missing={len(incompatible.missing_keys)}, '
          f'unexpected={len(incompatible.unexpected_keys)}')
    model.to(args.device)
    model.eval()
    return model


def static_kernel_from_solve(solve, h, w, device, dtype):
    weight_exp, k2 = solve.get_decay_map((h, w), device=device, dtype=dtype)
    tau = F.softplus(solve.raw_tau).clamp(max=4.0).to(device=device, dtype=dtype)
    rho = (torch.sigmoid(solve.raw_rho) * solve.rho_max).to(device=device, dtype=dtype)
    weight_exp = weight_exp.clamp_min(1e-6)
    kernel = rho + (1.0 - rho) * torch.exp(
        tau * torch.log(weight_exp[None, :, :, None]))
    return kernel.detach(), k2.detach()


def install_effective_mask_recorder(model, records):
    for stage_idx, layer in enumerate(model.layers):
        for block_idx, block in enumerate(layer.blocks):
            block.op._adaptive_stage = stage_idx + 1
            block.op._adaptive_is_last = block_idx == len(layer.blocks) - 1

            original_forward = block.op.forward

            def make_dno_forward(original):
                def dno_forward(self, x, compute_spectral_state=True):
                    y, spectral_state = original(x, compute_spectral_state)
                    rec = getattr(self.solve, '_adaptive_last_record', None)
                    if rec is not None and self._adaptive_is_last:
                        rec['stage'] = self._adaptive_stage
                        records.append(rec)
                    self.solve._adaptive_last_record = None
                    return y, spectral_state
                return dno_forward

            block.op.forward = MethodType(make_dno_forward(original_forward), block.op)

            solve = block.op.solve
            original_solve_forward = solve.forward

            def make_solve_forward(original):
                def solve_forward(self, x, param, compute_spectral_state=True):
                    y, spectral_state = original(x, param, compute_spectral_state)
                    if getattr(self, '_adaptive_parent_is_last', False):
                        _b, _c, h, w = x.shape
                        kernel, k2 = static_kernel_from_solve(self, h, w, x.device, x.dtype)
                        effective = kernel * param
                        self._adaptive_last_record = {
                            'effective': effective.detach().float().cpu(),
                            'param': param.detach().float().cpu(),
                            'k2': k2.detach().float().cpu(),
                        }
                    return y, spectral_state
                return solve_forward

            solve._adaptive_parent_is_last = block_idx == len(layer.blocks) - 1
            solve.forward = MethodType(make_solve_forward(original_solve_forward), solve)


def make_loader(args):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
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


def normalize_batch(x):
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def make_perturbations(images, args):
    images = images.clamp(0.0, 1.0)
    blur_kernel = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1
    blurred = TVF.gaussian_blur(images, kernel_size=[blur_kernel, blur_kernel],
                                sigma=[args.blur_sigma, args.blur_sigma])
    noise = (images + args.noise_std * torch.randn_like(images)).clamp(0.0, 1.0)
    highpass = images - blurred
    edge = (images + args.edge_strength * highpass).clamp(0.0, 1.0)
    return {
        'original': images,
        'blur': blurred,
        'noise': noise,
        'edge': edge,
    }


def effective_metrics(rec, perturbation, global_indices):
    effective = rec['effective'].numpy()  # B,H,W,C
    k2 = rec['k2'].numpy()
    response = effective.mean(axis=3)
    k2_max = float(k2.max()) if float(k2.max()) > 0 else 1.0
    low_mask = k2 <= 0.5 * k2_max
    high_mask = ~low_mask
    rows = []
    for i in range(response.shape[0]):
        r = response[i]
        total = float(r.sum()) + 1e-12
        low = float(r[low_mask].sum()) / total
        high = float(r[high_mask].sum()) / total
        centroid = float((r * k2).sum()) / total / k2_max
        rows.append({
            'image_index': int(global_indices[i]),
            'perturbation': perturbation,
            'stage': int(rec['stage']),
            'low_ratio': low,
            'high_ratio': high,
            'freq_centroid': centroid,
            'param_mean': float(rec['param'][i].mean()),
            'effective_mean': float(rec['effective'][i].mean()),
        })
    return rows


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['perturbation'], row['stage'])].append(row)
    summary = []
    for (perturbation, stage), vals in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        item = {
            'perturbation': perturbation,
            'stage': stage,
            'num_samples': len(vals),
        }
        for key in ('low_ratio', 'high_ratio', 'freq_centroid', 'param_mean', 'effective_mean'):
            arr = np.array([v[key] for v in vals], dtype=np.float64)
            item[f'{key}_mean'] = float(arr.mean())
            item[f'{key}_std'] = float(arr.std())
        summary.append(item)
    return summary


def summarize_delta(rows):
    by_key = {}
    for row in rows:
        by_key[(row['image_index'], row['stage'], row['perturbation'])] = row
    deltas = []
    for (image_index, stage, perturbation), row in by_key.items():
        if perturbation == 'original':
            continue
        base = by_key.get((image_index, stage, 'original'))
        if base is None:
            continue
        delta = {
            'image_index': image_index,
            'stage': stage,
            'perturbation': perturbation,
        }
        for key in ('low_ratio', 'high_ratio', 'freq_centroid'):
            delta[f'delta_{key}'] = row[key] - base[key]
        deltas.append(delta)

    grouped = defaultdict(list)
    for row in deltas:
        grouped[(row['perturbation'], row['stage'])].append(row)
    summary = []
    for (perturbation, stage), vals in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        item = {'perturbation': perturbation, 'stage': stage, 'num_samples': len(vals)}
        for key in ('delta_low_ratio', 'delta_high_ratio', 'delta_freq_centroid'):
            arr = np.array([v[key] for v in vals], dtype=np.float64)
            item[f'{key}_mean'] = float(arr.mean())
            item[f'{key}_std'] = float(arr.std())
        summary.append(item)
    return deltas, summary


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, delta_summary, output_dir, dpi):
    perturbations = ['original', 'blur', 'noise', 'edge']
    colors = {
        'original': '#4C72B0',
        'blur': '#55A868',
        'noise': '#C44E52',
        'edge': '#8172B2',
    }
    stages = sorted({row['stage'] for row in summary})
    labels = [f'S{stage}' for stage in stages]
    x = np.arange(len(stages))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=dpi)
    for perturbation in perturbations:
        rows = [r for r in summary if r['perturbation'] == perturbation]
        rows = sorted(rows, key=lambda r: r['stage'])
        axes[0].plot(x, [r['low_ratio_mean'] for r in rows], marker='o',
                     color=colors[perturbation], label=perturbation)
        axes[1].plot(x, [r['high_ratio_mean'] for r in rows], marker='o',
                     color=colors[perturbation], label=perturbation)
        axes[2].plot(x, [r['freq_centroid_mean'] for r in rows], marker='o',
                     color=colors[perturbation], label=perturbation)

    axes[0].set_title('Low-frequency response ratio')
    axes[1].set_title('High-frequency response ratio')
    axes[2].set_title('Normalized frequency centroid')
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[2].set_ylim(0.0, max(0.35, axes[2].get_ylim()[1]))
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'dno_perturbation_response.png', bbox_inches='tight')
    plt.close(fig)

    delta_perturbations = ['blur', 'noise', 'edge']
    width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=dpi)
    for idx, perturbation in enumerate(delta_perturbations):
        rows = [r for r in delta_summary if r['perturbation'] == perturbation]
        rows = sorted(rows, key=lambda r: r['stage'])
        offset = (idx - 1) * width
        axes[0].bar(x + offset, [r['delta_low_ratio_mean'] for r in rows],
                    width, color=colors[perturbation], label=perturbation)
        axes[1].bar(x + offset, [r['delta_high_ratio_mean'] for r in rows],
                    width, color=colors[perturbation], label=perturbation)
        axes[2].bar(x + offset, [r['delta_freq_centroid_mean'] for r in rows],
                    width, color=colors[perturbation], label=perturbation)
    axes[0].set_title('Delta low ratio vs original')
    axes[1].set_title('Delta high ratio vs original')
    axes[2].set_title('Delta frequency centroid vs original')
    for ax in axes:
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'dno_perturbation_delta.png', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    records = []
    install_effective_mask_recorder(model, records)
    loader, num_images = make_loader(args)
    print(f'using {num_images} images from {args.data_root}')

    all_rows = []
    seen = 0
    use_amp = args.amp and args.device.startswith('cuda')
    with torch.no_grad():
        for batch_idx, (images, _targets) in enumerate(loader):
            batch_size = images.shape[0]
            global_indices = list(range(seen, seen + batch_size))
            seen += batch_size
            images = images.to(args.device, non_blocking=True)
            perturbations = make_perturbations(images, args)
            for name, perturbed in perturbations.items():
                records.clear()
                normalized = normalize_batch(perturbed)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _ = model(normalized)
                for rec in records:
                    all_rows.extend(effective_metrics(rec, name, global_indices))
            if batch_idx % 5 == 0:
                print(f'processed {seen}/{num_images}')

    summary = summarize_rows(all_rows)
    deltas, delta_summary = summarize_delta(all_rows)

    write_csv(output_dir / 'dno_perturbation_samples.csv', all_rows)
    write_csv(output_dir / 'dno_perturbation_summary.csv', summary)
    write_csv(output_dir / 'dno_perturbation_delta_samples.csv', deltas)
    write_csv(output_dir / 'dno_perturbation_delta_summary.csv', delta_summary)
    (output_dir / 'dno_perturbation_summary.json').write_text(
        json.dumps({
            'data_root': args.data_root,
            'checkpoint': args.checkpoint,
            'max_images': args.max_images,
            'blur_kernel': args.blur_kernel,
            'blur_sigma': args.blur_sigma,
            'noise_std': args.noise_std,
            'edge_strength': args.edge_strength,
            'summary': summary,
            'delta_summary': delta_summary,
        }, indent=2),
        encoding='utf-8',
    )
    plot_summary(summary, delta_summary, output_dir, args.dpi)
    print(f'saved DNO adaptive perturbation statistics to {output_dir}')


if __name__ == '__main__':
    main()
