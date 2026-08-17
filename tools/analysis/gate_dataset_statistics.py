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
        description='Compute dataset-level statistics for VPF gate behavior.')
    parser.add_argument(
        '--data-root',
        default='/data1/sunmy/Imagenet1k/data/val',
        help='ImageFolder root. Use ImageNet val by default.')
    parser.add_argument(
        '--checkpoint',
        default='/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth',
        help='VPF checkpoint path.')
    parser.add_argument(
        '--output-dir',
        default=str(CLASSIFICATION_ROOT / 'analysis' / 'gate_outputs_test' / 'dataset_mean'),
        help='Output directory for summary files.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--depths', default='2,3,4,2')
    parser.add_argument('--dims', type=int, default=96)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=0, help='0 means all images.')
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument(
        '--save-per-image',
        action='store_true',
        help='Save per-image per-stage statistics. Val set is usually still below 100 MB.')
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


def install_gate_recorder(model, records):
    def make_forward(layer, stage_index):
        def forward(self, x):
            spectral_state = None
            last_block_index = len(self.blocks) - 1
            for block_index, block in enumerate(self.blocks):
                x, spectral_state = block(
                    x,
                    compute_spectral_state=(
                        self.use_galerkin and self.use_gate and
                        block_index == last_block_index),
                )
            if self.use_galerkin:
                global_delta = self.global_branch(x)
                if self.use_gate:
                    feature_state = x.mean(dim=(2, 3))
                    feature_embed = self.feature_encoder(feature_state)
                    spectral_embed = self.spectral_encoder(spectral_state)
                    gate = self.joint_gate(torch.cat([feature_embed, spectral_embed], dim=-1))
                    alpha = self.global_alpha[None, :, None, None]
                    gated_delta = alpha * gate[:, :, None, None] * global_delta
                    x = x + gated_delta
                    records.append({
                        'stage': stage_index + 1,
                        'gate': gate.detach(),
                        'alpha': self.global_alpha.detach(),
                        'global_delta': global_delta.detach(),
                        'gated_delta': gated_delta.detach(),
                    })
                else:
                    x = x + self.global_alpha[None, :, None, None] * global_delta
            x = self.downsample(x)
            return x
        return MethodType(forward, layer)

    for stage_index, layer in enumerate(model.layers):
        if getattr(layer, 'use_galerkin', False) and getattr(layer, 'use_gate', False):
            layer.forward = make_forward(layer, stage_index)


class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = defaultdict(float)
        self.sumsq = defaultdict(float)

    def update(self, values):
        for item in values:
            self.n += 1
            for key, value in item.items():
                self.sum[key] += float(value)
                self.sumsq[key] += float(value) * float(value)

    def mean(self, key):
        return self.sum[key] / max(self.n, 1)

    def std(self, key):
        if self.n <= 1:
            return 0.0
        mean = self.mean(key)
        var = self.sumsq[key] / self.n - mean * mean
        return max(var, 0.0) ** 0.5


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


def summarize_records(records, topk):
    rows = []
    for record in records:
        gate = record['gate'].float()
        alpha = record['alpha'].float()
        global_delta = record['global_delta'].float()
        gated_delta = record['gated_delta'].float()
        channel_score = gate * alpha.abs()[None, :]
        k = min(topk, channel_score.shape[1])
        topk_sum = channel_score.topk(k, dim=1).values.sum(dim=1)
        total_sum = channel_score.sum(dim=1).clamp_min(1e-12)
        topk_ratio = topk_sum / total_sum

        batch_rows = {
            'stage': record['stage'],
            'gate_mean': gate.mean(dim=1),
            'gate_std_channel': gate.std(dim=1, unbiased=False),
            'gate_min': gate.min(dim=1).values,
            'gate_max': gate.max(dim=1).values,
            'alpha_abs_mean': alpha.abs().mean().expand(gate.shape[0]),
            'global_delta_abs_mean': global_delta.abs().mean(dim=(1, 2, 3)),
            'gated_delta_abs_mean': gated_delta.abs().mean(dim=(1, 2, 3)),
            'topk_channel_ratio': topk_ratio,
        }
        for i in range(gate.shape[0]):
            row = {key: value[i].item() if torch.is_tensor(value) else value
                   for key, value in batch_rows.items()}
            row['sample_index_in_batch'] = i
            rows.append(row)
    return rows


def save_summary_csv(stats_by_stage, output_path):
    keys = [
        'gate_mean', 'gate_std_channel', 'gate_min', 'gate_max',
        'alpha_abs_mean', 'global_delta_abs_mean', 'gated_delta_abs_mean',
        'topk_channel_ratio',
    ]
    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'num_images'] + [f'{k}_mean' for k in keys] +
                        [f'{k}_std_over_images' for k in keys])
        for stage in sorted(stats_by_stage):
            stat = stats_by_stage[stage]
            writer.writerow(
                [stage, stat.n] +
                [f'{stat.mean(k):.8f}' for k in keys] +
                [f'{stat.std(k):.8f}' for k in keys]
            )


def save_per_image_csv(rows, output_path):
    if not rows:
        return
    fieldnames = ['image_index'] + list(rows[0].keys())
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(stats_by_stage, output_path, args, num_images):
    keys = [
        'gate_mean', 'gate_std_channel', 'gate_min', 'gate_max',
        'alpha_abs_mean', 'global_delta_abs_mean', 'gated_delta_abs_mean',
        'topk_channel_ratio',
    ]
    payload = {
        'data_root': args.data_root,
        'checkpoint': args.checkpoint,
        'num_images': num_images,
        'topk': args.topk,
        'stages': {},
    }
    for stage in sorted(stats_by_stage):
        stat = stats_by_stage[stage]
        payload['stages'][str(stage)] = {
            'num_images': stat.n,
            **{f'{k}_mean': stat.mean(k) for k in keys},
            **{f'{k}_std_over_images': stat.std(k) for k in keys},
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def save_plots(stats_by_stage, output_dir, dpi):
    stages = sorted(stats_by_stage)
    labels = [f'S{stage}' for stage in stages]

    def means(key):
        return [stats_by_stage[stage].mean(key) for stage in stages]

    def stds(key):
        return [stats_by_stage[stage].std(key) for stage in stages]

    x = np.arange(len(stages))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=dpi)

    axes[0].bar(x, means('gate_mean'), yerr=stds('gate_mean'), capsize=3)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title('Gate mean')

    width = 0.36
    axes[1].bar(x - width / 2, means('global_delta_abs_mean'), width=width,
                label='|global delta|')
    axes[1].bar(x + width / 2, means('gated_delta_abs_mean'), width=width,
                label='|alpha*gate*delta|')
    axes[1].set_xticks(x, labels)
    axes[1].set_title('Global injection')
    axes[1].legend(frameon=False)

    axes[2].bar(x, means('topk_channel_ratio'), yerr=stds('topk_channel_ratio'), capsize=3)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title('Top-k channel concentration')

    fig.tight_layout()
    fig.savefig(output_dir / 'gate_dataset_summary.png', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, num_images = make_loader(args)
    model = build_model(args)
    records = []
    install_gate_recorder(model, records)

    stats_by_stage = defaultdict(RunningStats)
    per_image_rows = []
    seen = 0
    with torch.no_grad():
        for batch_idx, (images, _targets) in enumerate(loader):
            records.clear()
            images = images.to(args.device, non_blocking=True)
            _ = model(images)
            rows = summarize_records(records, args.topk)
            batch_size = images.shape[0]
            for row in rows:
                stage = int(row['stage'])
                stats_by_stage[stage].update([row])
            if args.save_per_image:
                for row in rows:
                    row = dict(row)
                    row['image_index'] = seen + int(row.pop('sample_index_in_batch'))
                    per_image_rows.append(row)
            seen += batch_size
            if batch_idx % 20 == 0:
                print(f'processed {seen}/{num_images}')

    save_summary_csv(stats_by_stage, output_dir / 'gate_dataset_summary.csv')
    save_json(stats_by_stage, output_dir / 'gate_dataset_summary.json', args, num_images)
    save_plots(stats_by_stage, output_dir, args.dpi)
    if args.save_per_image:
        save_per_image_csv(per_image_rows, output_dir / 'gate_per_image_stats.csv')

    print(f'saved summaries to {output_dir}')


if __name__ == '__main__':
    main()
