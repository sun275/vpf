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
REPO_ROOT = CLASSIFICATION_ROOT.parent
sys.path.insert(0, str(CLASSIFICATION_ROOT))

from vpf.models import vpf  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_int_tuple(text):
    return tuple(int(v.strip()) for v in text.split(',') if v.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare DNO and gated global branch activations in VPF.')
    parser.add_argument('--image', default=None, help='Single image path for contribution maps.')
    parser.add_argument('--image-dir', default=None, help='Optional directory for multiple image maps.')
    parser.add_argument(
        '--data-root',
        default=None,
        help='ImageFolder root for dataset-level statistics. Example: ImageNet val root.')
    parser.add_argument(
        '--checkpoint',
        default='/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth')
    parser.add_argument(
        '--output-dir',
        default=str(CLASSIFICATION_ROOT / 'analysis' / 'branch_outputs'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--depths', default='2,3,4,2')
    parser.add_argument('--dims', type=int, default=96)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=0, help='0 means all images.')
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--amp', action='store_true')
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


def install_branch_recorder(model, records):
    for stage_index, layer in enumerate(model.layers):
        layer._branch_stage_index = stage_index + 1
        layer._branch_dno_deltas = []

        for block in layer.blocks:
            block._branch_parent_layer = layer

            def block_forward(self, input_tensor, compute_spectral_state=True):
                x = input_tensor
                if not self.layer_scale:
                    if self.post_norm:
                        op_out, spectral_state = self.op(x, compute_spectral_state)
                        dno_delta = self.drop_path(self.norm1(op_out))
                        x = x + dno_delta
                        if self.mlp_branch:
                            x = x + self.drop_path(self.norm2(self.mlp(x)))
                    else:
                        op_out, spectral_state = self.op(self.norm1(x), compute_spectral_state)
                        dno_delta = self.drop_path(op_out)
                        x = x + dno_delta
                        if self.mlp_branch:
                            x = x + self.drop_path(self.mlp(self.norm2(x)))
                else:
                    if self.post_norm:
                        op_out, spectral_state = self.op(x, compute_spectral_state)
                        dno_delta = self.drop_path(
                            self.gamma1[:, None, None] * self.norm1(op_out))
                        x = x + dno_delta
                        if self.mlp_branch:
                            x = x + self.drop_path(
                                self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
                    else:
                        op_out, spectral_state = self.op(self.norm1(x), compute_spectral_state)
                        dno_delta = self.drop_path(self.gamma1[:, None, None] * op_out)
                        x = x + dno_delta
                        if self.mlp_branch:
                            x = x + self.drop_path(
                                self.gamma2[:, None, None] * self.mlp(self.norm2(x)))

                self._branch_parent_layer._branch_dno_deltas.append(
                    dno_delta.detach().float().cpu())
                return x, spectral_state

            block.forward = MethodType(block_forward, block)

        def layer_forward(self, x):
            self._branch_dno_deltas = []
            spectral_state = None
            last_block_index = len(self.blocks) - 1
            for block_index, block in enumerate(self.blocks):
                x, spectral_state = block(
                    x,
                    compute_spectral_state=(
                        self.use_galerkin and self.use_gate and
                        block_index == last_block_index),
                )

            global_delta = None
            global_injection = None
            gate = None
            if self.use_galerkin:
                global_delta = self.global_branch(x)
                if self.use_gate:
                    feature_state = x.mean(dim=(2, 3))
                    feature_embed = self.feature_encoder(feature_state)
                    spectral_embed = self.spectral_encoder(spectral_state)
                    gate = self.joint_gate(torch.cat([feature_embed, spectral_embed], dim=-1))
                    global_injection = (
                        self.global_alpha[None, :, None, None] *
                        gate[:, :, None, None] * global_delta)
                    x = x + global_injection
                else:
                    global_injection = self.global_alpha[None, :, None, None] * global_delta
                    x = x + global_injection

            dno_deltas = torch.stack(self._branch_dno_deltas, dim=0)
            records.append({
                'stage': self._branch_stage_index,
                'dno_deltas': dno_deltas,
                'dno_abs_map': dno_deltas.abs().mean(dim=(0, 2)),
                'dno_abs_mean': dno_deltas.abs().mean(dim=(0, 2, 3, 4)),
                'global_delta': None if global_delta is None else global_delta.detach().float().cpu(),
                'global_injection': None if global_injection is None else global_injection.detach().float().cpu(),
                'gate': None if gate is None else gate.detach().float().cpu(),
            })

            x = self.downsample(x)
            return x

        layer.forward = MethodType(layer_forward, layer)


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
    tensor = transform(image).unsqueeze(0)
    display = np.asarray(display_transform(image)).astype(np.float32) / 255.0
    return tensor, display


def normalize_map(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    return arr


def upsample_map(map_tensor, img_size):
    if map_tensor.ndim == 2:
        map_tensor = map_tensor[None]
    if map_tensor.ndim == 3:
        map_tensor = map_tensor[:, None]
    up = F.interpolate(map_tensor, size=(img_size, img_size),
                       mode='bilinear', align_corners=False)
    return up[:, 0].numpy()


def overlay_map(image, heat, alpha):
    cmap = plt.get_cmap('jet')(heat)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * cmap, 0.0, 1.0)


def record_maps(record, img_size):
    dno_maps = upsample_map(record['dno_abs_map'], img_size)
    dno_map = normalize_map(dno_maps[0])
    if record['global_injection'] is None:
        global_map = np.zeros_like(dno_map)
    else:
        fmap = record['global_injection'].abs().mean(dim=1)
        global_map = normalize_map(upsample_map(fmap, img_size)[0])
    return dno_map, global_map


def save_single_image_maps(image_path, model, args):
    records = []
    install_branch_recorder(model, records)
    tensor, display = load_image(image_path, args.img_size)
    tensor = tensor.to(args.device)
    use_amp = args.amp and args.device.startswith('cuda')
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(tensor)
            prob = logits.float().softmax(dim=1)
            pred_prob, pred_class = prob.max(dim=1)

    out_dir = Path(args.output_dir) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3.0 * n), dpi=args.dpi)
    if n == 1:
        axes = axes[None]
    for row, record in enumerate(records):
        dno_map, global_map = record_maps(record, args.img_size)
        ratio = branch_ratio(record)
        axes[row, 0].imshow(display)
        axes[row, 0].set_title(f'Stage {record["stage"]} input')
        axes[row, 1].imshow(dno_map, cmap='jet', vmin=0.0, vmax=1.0)
        axes[row, 1].set_title('DNO contribution')
        axes[row, 2].imshow(overlay_map(display, dno_map, args.alpha))
        axes[row, 2].set_title('DNO overlay')
        axes[row, 3].imshow(global_map, cmap='jet', vmin=0.0, vmax=1.0)
        axes[row, 3].set_title(f'Global injection\nratio={ratio:.3f}')
        axes[row, 4].imshow(overlay_map(display, global_map, args.alpha))
        axes[row, 4].set_title('Global overlay')
        for col in range(5):
            axes[row, col].axis('off')
    fig.suptitle(
        f'{image_path.name} | class={int(pred_class.item())}, '
        f'p={float(pred_prob.item()):.3f}', y=1.002)
    fig.tight_layout()
    fig.savefig(out_dir / 'branch_activation_comparison.png', bbox_inches='tight')
    plt.close(fig)

    with (out_dir / 'branch_activation_stats.csv').open('w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['stage', 'dno_abs_mean', 'global_injection_abs_mean',
                           'global_delta_abs_mean', 'global_to_dno_ratio'])
        writer.writeheader()
        for record in records:
            writer.writerow(record_summary(record))
    print(f'saved image maps to {out_dir}')


def branch_ratio(record):
    dno = float(record['dno_abs_mean'].mean().item())
    if record['global_injection'] is None:
        return 0.0
    glob = float(record['global_injection'].abs().mean().item())
    return glob / max(dno, 1e-12)


def record_summary(record):
    dno = float(record['dno_abs_mean'].mean().item())
    if record['global_injection'] is None:
        glob = 0.0
        raw = 0.0
    else:
        glob = float(record['global_injection'].abs().mean().item())
        raw = float(record['global_delta'].abs().mean().item())
    return {
        'stage': record['stage'],
        'dno_abs_mean': dno,
        'global_injection_abs_mean': glob,
        'global_delta_abs_mean': raw,
        'global_to_dno_ratio': glob / max(dno, 1e-12),
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


def compute_dataset_stats(model, args):
    if not args.data_root:
        return
    records = []
    install_branch_recorder(model, records)
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
            batch_size = images.shape[0]
            for record in records:
                dno_per_sample = record['dno_abs_mean'].numpy()
                if record['global_injection'] is None:
                    glob_per_sample = np.zeros_like(dno_per_sample)
                    raw_per_sample = np.zeros_like(dno_per_sample)
                else:
                    glob_per_sample = record['global_injection'].abs().mean(dim=(1, 2, 3)).numpy()
                    raw_per_sample = record['global_delta'].abs().mean(dim=(1, 2, 3)).numpy()
                for i in range(batch_size):
                    dno = float(dno_per_sample[i])
                    glob = float(glob_per_sample[i])
                    raw = float(raw_per_sample[i])
                    stats_by_stage[record['stage']].update({
                        'stage': record['stage'],
                        'dno_abs_mean': dno,
                        'global_injection_abs_mean': glob,
                        'global_delta_abs_mean': raw,
                        'global_to_dno_ratio': glob / max(dno, 1e-12),
                    })
            seen += batch_size
            if batch_idx % 20 == 0:
                print(f'processed {seen}/{num_images}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ['dno_abs_mean', 'global_injection_abs_mean',
            'global_delta_abs_mean', 'global_to_dno_ratio']
    rows = []
    for stage in sorted(stats_by_stage):
        stat = stats_by_stage[stage]
        row = {'stage': stage, 'num_images': stat.n}
        for key in keys:
            row[f'{key}_mean'] = stat.mean(key)
            row[f'{key}_std'] = stat.std(key)
        rows.append(row)

    with (output_dir / 'branch_dataset_summary.csv').open('w', newline='') as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / 'branch_dataset_summary.json').write_text(
        json.dumps({'data_root': args.data_root, 'checkpoint': args.checkpoint,
                    'rows': rows}, indent=2),
        encoding='utf-8')
    save_dataset_plot(rows, output_dir, args.dpi)
    print(f'saved dataset stats to {output_dir}')


def save_dataset_plot(rows, output_dir, dpi):
    stages = [row['stage'] for row in rows]
    labels = [f'S{stage}' for stage in stages]
    x = np.arange(len(rows))
    width = 0.34

    dno = [row['dno_abs_mean_mean'] for row in rows]
    glob = [row['global_injection_abs_mean_mean'] for row in rows]
    ratio = [row['global_to_dno_ratio_mean'] for row in rows]
    raw = [row['global_delta_abs_mean_mean'] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=dpi)
    axes[0].bar(x - width / 2, dno, width=width, label='DNO')
    axes[0].bar(x + width / 2, glob, width=width, label='Gated global')
    axes[0].set_xticks(x, labels)
    axes[0].set_title('Effective branch strength')
    axes[0].legend(frameon=False)

    axes[1].bar(x, ratio)
    axes[1].set_xticks(x, labels)
    axes[1].set_title('Global / DNO ratio')

    axes[2].bar(x - width / 2, raw, width=width, label='Raw global')
    axes[2].bar(x + width / 2, glob, width=width, label='Gated global')
    axes[2].set_xticks(x, labels)
    axes[2].set_title('Gate regulation')
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / 'branch_dataset_summary.png', bbox_inches='tight')
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


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args)
    if image_paths:
        for image_path in image_paths:
            model = build_model(args)
            save_single_image_maps(image_path, model, args)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.data_root:
        model = build_model(args)
        compute_dataset_stats(model, args)


if __name__ == '__main__':
    main()
