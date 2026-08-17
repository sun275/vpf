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
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


THIS_DIR = Path(__file__).resolve().parent
CLASSIFICATION_ROOT = THIS_DIR.parent
sys.path.insert(0, str(CLASSIFICATION_ROOT))

from vpf.models import vpf  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_int_tuple(text):
    return tuple(int(v.strip()) for v in text.split(',') if v.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize static/dynamic spectral response and DNO frequency energy.')
    parser.add_argument('--image', default=None, help='Single image path.')
    parser.add_argument('--image-dir', default=None, help='Optional image directory.')
    parser.add_argument('--data-root', default=None, help='ImageFolder root for dataset statistics.')
    parser.add_argument(
        '--checkpoint',
        default='/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth')
    parser.add_argument(
        '--output-dir',
        default=str(CLASSIFICATION_ROOT / 'analysis' / 'spectral_outputs'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--depths', default='2,3,4,2')
    parser.add_argument('--dims', type=int, default=96)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
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


def install_recorders(model, records):
    for stage_idx, layer in enumerate(model.layers):
        for block_idx, block in enumerate(layer.blocks):
            block.op._spectral_stage = stage_idx + 1
            block.op._spectral_block = block_idx
            block.op._spectral_is_last = block_idx == len(layer.blocks) - 1
            original_forward = block.op.forward

            def make_dno_forward(original):
                def dno_forward(self, x, compute_spectral_state=True):
                    y, spectral_state = original(x, compute_spectral_state)
                    if self._spectral_is_last:
                        energy_in = frequency_energy_ratio(x.detach().float())
                        energy_out = frequency_energy_ratio(y.detach().float())
                        rec = getattr(self.solve, '_last_spectral_record', None)
                        if rec is not None:
                            rec.update({
                                'stage': self._spectral_stage,
                                'block': self._spectral_block,
                                'energy_in': energy_in,
                                'energy_out': energy_out,
                            })
                            records.append(rec)
                            self.solve._last_spectral_record = None
                    return y, spectral_state
                return dno_forward

            block.op.forward = MethodType(make_dno_forward(original_forward), block.op)

            solve = block.op.solve
            original_solve_forward = solve.forward

            def make_solve_forward(original):
                def solve_forward(self, x, param, compute_spectral_state=True):
                    y, spectral_state = original(x, param, compute_spectral_state)
                    if getattr(self, '_spectral_parent_is_last', False) or compute_spectral_state:
                        _b, _c, h, w = x.shape
                        kernel, k2 = static_kernel_from_solve(self, h, w, x.device, x.dtype)
                        effective = kernel * param
                        self._last_spectral_record = {
                            'kernel': kernel.detach().float().cpu(),
                            'effective': effective.detach().float().cpu(),
                            'param': param.detach().float().cpu(),
                            'k2': k2.detach().float().cpu(),
                        }
                    return y, spectral_state
                return solve_forward

            solve._spectral_parent_is_last = block_idx == len(layer.blocks) - 1
            solve.forward = MethodType(make_solve_forward(original_solve_forward), solve)


def frequency_energy_ratio(x):
    # x: B,C,H,W. Return per-sample low/mid/high ratios.
    x = x.float()
    fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
    energy = fft.abs().square().mean(dim=1)
    _b, h, w = energy.shape
    yy = torch.arange(h, device=x.device) - (h - 1) / 2
    xx = torch.arange(w, device=x.device) - (w - 1) / 2
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    r = torch.sqrt(gx.square() + gy.square())
    rmax = r.max().clamp_min(1.0)
    low = r <= rmax / 3
    mid = (r > rmax / 3) & (r <= 2 * rmax / 3)
    high = r > 2 * rmax / 3
    total = energy.sum(dim=(1, 2)).clamp_min(1e-12)
    return {
        'low': (energy[:, low].sum(dim=1) / total).detach().cpu(),
        'mid': (energy[:, mid].sum(dim=1) / total).detach().cpu(),
        'high': (energy[:, high].sum(dim=1) / total).detach().cpu(),
    }


def load_image(path, img_size):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    display_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
    ])
    return transform(image).unsqueeze(0), np.asarray(display_transform(image)).astype(np.float32) / 255.0


def normalize_map(arr, log_scale=False):
    arr = arr.astype(np.float32)
    if log_scale:
        arr = np.log1p(np.maximum(arr, 0.0))
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    return arr


def mean_map(tensor):
    # B,H,W,C or 1,H,W,C -> H,W average over batch/channel.
    arr = tensor.numpy()
    return arr.mean(axis=(0, 3))


def radial_profile(response, k2, bins=24):
    # response: H,W, k2: H,W
    r = np.sqrt(k2.reshape(-1))
    y = response.reshape(-1)
    edges = np.linspace(float(r.min()), float(r.max()), bins + 1)
    values = []
    centers = []
    for i in range(bins):
        if i == bins - 1:
            mask = (r >= edges[i]) & (r <= edges[i + 1])
        else:
            mask = (r >= edges[i]) & (r < edges[i + 1])
        centers.append((edges[i] + edges[i + 1]) / 2)
        values.append(float(y[mask].mean()) if mask.any() else 0.0)
    return np.array(centers), np.array(values)


def save_image_spectral(records, image_path, display, logits, args):
    out_dir = Path(args.output_dir) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    prob = logits.float().softmax(dim=1)
    pred_prob, pred_class = prob.max(dim=1)

    records = sorted(records, key=lambda r: r['stage'])
    fig, axes = plt.subplots(3, len(records), figsize=(3.3 * len(records), 9.0), dpi=args.dpi)
    if len(records) == 1:
        axes = axes[:, None]
    for col, rec in enumerate(records):
        static = mean_map(rec['kernel'])
        dynamic = mean_map(rec['effective'])
        param = mean_map(rec['param'])
        axes[0, col].imshow(normalize_map(static), cmap='magma')
        axes[0, col].set_title(f'S{rec["stage"]} static kernel')
        axes[1, col].imshow(normalize_map(dynamic), cmap='magma')
        axes[1, col].set_title('dynamic kernel*param')
        axes[2, col].imshow(normalize_map(param), cmap='magma')
        axes[2, col].set_title('dynamic param')
        for row in range(3):
            axes[row, col].axis('off')
    fig.suptitle(
        f'{image_path.name} | class={int(pred_class.item())}, p={float(pred_prob.item()):.3f}',
        y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'spectral_response_maps.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=args.dpi)
    for rec in records:
        static = mean_map(rec['kernel'])
        dynamic = mean_map(rec['effective'])
        k2 = rec['k2'].numpy()
        x_static, y_static = radial_profile(static, k2)
        _x_dynamic, y_dynamic = radial_profile(dynamic, k2)
        ax.plot(x_static, y_static, label=f'S{rec["stage"]} static')
        ax.plot(x_static, y_dynamic, '--', label=f'S{rec["stage"]} dynamic')
    ax.set_xlabel('frequency radius')
    ax.set_ylabel('mean response')
    ax.set_title('Radial spectral response')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'spectral_radial_profile.png', bbox_inches='tight')
    plt.close(fig)

    with (out_dir / 'spectral_energy_stats.csv').open('w', newline='') as f:
        fieldnames = [
            'stage', 'input_low', 'input_mid', 'input_high',
            'output_low', 'output_mid', 'output_high',
            'static_mean', 'dynamic_mean', 'param_mean',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(record_energy_row(rec))
    print(f'saved spectral maps to {out_dir}')


def record_energy_row(rec):
    return {
        'stage': rec['stage'],
        'input_low': float(rec['energy_in']['low'][0]),
        'input_mid': float(rec['energy_in']['mid'][0]),
        'input_high': float(rec['energy_in']['high'][0]),
        'output_low': float(rec['energy_out']['low'][0]),
        'output_mid': float(rec['energy_out']['mid'][0]),
        'output_high': float(rec['energy_out']['high'][0]),
        'static_mean': float(rec['kernel'].mean()),
        'dynamic_mean': float(rec['effective'].mean()),
        'param_mean': float(rec['param'].mean()),
    }


class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = defaultdict(float)
        self.sumsq = defaultdict(float)

    def update(self, row):
        self.n += 1
        for key, value in row.items():
            if key == 'stage':
                continue
            self.sum[key] += float(value)
            self.sumsq[key] += float(value) * float(value)

    def mean(self, key):
        return self.sum[key] / max(self.n, 1)

    def std(self, key):
        if self.n <= 1:
            return 0.0
        mean = self.mean(key)
        return max(self.sumsq[key] / self.n - mean * mean, 0.0) ** 0.5


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
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return loader, len(dataset)


def summarize_batch_records(records):
    rows = []
    for rec in records:
        batch_size = rec['effective'].shape[0]
        static_mean = rec['kernel'].mean().item()
        for i in range(batch_size):
            rows.append({
                'stage': rec['stage'],
                'input_low': float(rec['energy_in']['low'][i]),
                'input_mid': float(rec['energy_in']['mid'][i]),
                'input_high': float(rec['energy_in']['high'][i]),
                'output_low': float(rec['energy_out']['low'][i]),
                'output_mid': float(rec['energy_out']['mid'][i]),
                'output_high': float(rec['energy_out']['high'][i]),
                'static_mean': static_mean,
                'dynamic_mean': float(rec['effective'][i].mean()),
                'param_mean': float(rec['param'][i].mean()),
            })
    return rows


def compute_dataset_stats(model, args):
    records = []
    install_recorders(model, records)
    loader, num_images = make_loader(args)
    stats_by_stage = defaultdict(RunningStats)
    seen = 0
    use_amp = args.amp and args.device.startswith('cuda')
    with torch.no_grad():
        for batch_idx, (images, _targets) in enumerate(loader):
            records.clear()
            images = images.to(args.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _ = model(images)
            for row in summarize_batch_records(records):
                stats_by_stage[row['stage']].update(row)
            seen += images.shape[0]
            if batch_idx % 20 == 0:
                print(f'processed {seen}/{num_images}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        'input_low', 'input_mid', 'input_high',
        'output_low', 'output_mid', 'output_high',
        'static_mean', 'dynamic_mean', 'param_mean',
    ]
    rows = []
    for stage in sorted(stats_by_stage):
        stat = stats_by_stage[stage]
        row = {'stage': stage, 'num_images': stat.n}
        for key in keys:
            row[f'{key}_mean'] = stat.mean(key)
            row[f'{key}_std'] = stat.std(key)
        rows.append(row)

    with (output_dir / 'spectral_dataset_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / 'spectral_dataset_summary.json').write_text(
        json.dumps({'data_root': args.data_root, 'checkpoint': args.checkpoint,
                    'rows': rows}, indent=2),
        encoding='utf-8')
    save_dataset_plot(rows, output_dir, args.dpi)
    print(f'saved dataset spectral stats to {output_dir}')


def save_dataset_plot(rows, output_dir, dpi):
    stages = [row['stage'] for row in rows]
    labels = [f'S{stage}' for stage in stages]
    x = np.arange(len(rows))
    width = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=dpi)
    axes[0].bar(x - width, [r['input_low_mean'] for r in rows], width, label='input low')
    axes[0].bar(x, [r['input_mid_mean'] for r in rows], width, label='input mid')
    axes[0].bar(x + width, [r['input_high_mean'] for r in rows], width, label='input high')
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title('DNO input frequency energy')
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x - width, [r['output_low_mean'] for r in rows], width, label='output low')
    axes[1].bar(x, [r['output_mid_mean'] for r in rows], width, label='output mid')
    axes[1].bar(x + width, [r['output_high_mean'] for r in rows], width, label='output high')
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title('DNO output frequency energy')
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x - width / 2, [r['static_mean_mean'] for r in rows], width, label='static kernel')
    axes[2].bar(x + width / 2, [r['dynamic_mean_mean'] for r in rows], width, label='kernel*param')
    axes[2].set_xticks(x, labels)
    axes[2].set_title('Mean spectral response')
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / 'spectral_dataset_summary.png', bbox_inches='tight')
    plt.close(fig)


def collect_images(args):
    if args.image_dir:
        paths = [
            p for p in sorted(Path(args.image_dir).rglob('*'))
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if args.max_images > 0:
            paths = paths[:args.max_images]
        return paths
    if args.image:
        return [Path(args.image)]
    return []


def run_image(model, image_path, args):
    records = []
    install_recorders(model, records)
    tensor, display = load_image(image_path, args.img_size)
    tensor = tensor.to(args.device)
    use_amp = args.amp and args.device.startswith('cuda')
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(tensor)
    save_image_spectral(records, image_path, display, logits, args)


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args)
    if image_paths:
        for image_path in image_paths:
            model = build_model(args)
            run_image(model, image_path, args)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.data_root:
        model = build_model(args)
        compute_dataset_stats(model, args)


if __name__ == '__main__':
    main()
