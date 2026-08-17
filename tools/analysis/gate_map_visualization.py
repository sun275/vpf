#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


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
        description='Visualize VPF stage-wise gate strength and gated global contribution maps.')
    parser.add_argument(
        '--image',
        default=str(REPO_ROOT / 'visual' / 'ILSVRC2012_val_00000293.JPEG'),
        help='Input image path. Ignored when --image-dir is set.')
    parser.add_argument('--image-dir', default=None, help='Optional directory of images.')
    parser.add_argument(
        '--checkpoint',
        default='/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth',
        help='VPF checkpoint path.')
    parser.add_argument(
        '--output-dir',
        default=str(CLASSIFICATION_ROOT / 'analysis' / 'gate_outputs'),
        help='Directory for saved figures and csv files.')
    parser.add_argument('--device', default='cuda', help='cuda or cpu.')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--depths', default='2,3,4,2')
    parser.add_argument('--dims', type=int, default=96)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=0, help='0 means all images.')
    parser.add_argument('--alpha', type=float, default=0.45, help='Overlay opacity.')
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


def install_gate_recorder(model):
    records = []

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
                        'gate': gate.detach().float().cpu(),
                        'alpha': self.global_alpha.detach().float().cpu(),
                        'global_delta': global_delta.detach().float().cpu(),
                        'gated_delta': gated_delta.detach().float().cpu(),
                        'feature_state': feature_state.detach().float().cpu(),
                        'spectral_state': spectral_state.detach().float().cpu(),
                    })
                else:
                    x = x + self.global_alpha[None, :, None, None] * global_delta
            x = self.downsample(x)
            return x
        return MethodType(forward, layer)

    for stage_index, layer in enumerate(model.layers):
        if getattr(layer, 'use_galerkin', False) and getattr(layer, 'use_gate', False):
            layer.forward = make_forward(layer, stage_index)

    return records


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


def contribution_map(record, img_size, mode='gated'):
    key = 'gated_delta' if mode == 'gated' else 'global_delta'
    fmap = record[key].abs().mean(dim=1, keepdim=True)
    fmap = F.interpolate(fmap, size=(img_size, img_size), mode='bilinear', align_corners=False)
    return normalize_map(fmap[0, 0].numpy())


def overlay_map(image, heat, alpha):
    cmap = plt.get_cmap('jet')(heat)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * cmap, 0.0, 1.0)


def save_gate_bar(records, output_path, dpi):
    stages = [r['stage'] for r in records]
    gate_mean = [r['gate'].mean().item() for r in records]
    gate_std = [r['gate'].std().item() for r in records]
    alpha_abs_mean = [r['alpha'].abs().mean().item() for r in records]
    effect_mean = [r['gated_delta'].abs().mean().item() for r in records]

    x = np.arange(len(stages))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), dpi=dpi)
    axes[0].bar(x, gate_mean, yerr=gate_std, capsize=3, color='#4c78a8')
    axes[0].set_xticks(x, [f'S{stage}' for stage in stages])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel('gate mean')
    axes[0].set_title('Stage gate strength')

    width = 0.36
    axes[1].bar(x - width / 2, alpha_abs_mean, width=width, label='|alpha|')
    axes[1].bar(x + width / 2, effect_mean, width=width, label='|gated delta|')
    axes[1].set_xticks(x, [f'S{stage}' for stage in stages])
    axes[1].set_title('Injected global strength')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def save_stage_maps(records, image, output_path, img_size, alpha, dpi):
    n = len(records)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), dpi=dpi)
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, record in enumerate(records):
        heat = contribution_map(record, img_size, mode='gated')
        axes[0, col].imshow(heat, cmap='jet', vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f'Stage {record["stage"]} gated map')
        axes[1, col].imshow(overlay_map(image, heat, alpha))
        axes[1, col].set_title('overlay')
        axes[0, col].axis('off')
        axes[1, col].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def save_channel_maps(records, output_path, dpi, topk=32):
    fig, axes = plt.subplots(len(records), 1, figsize=(8, 2.1 * len(records)), dpi=dpi)
    if len(records) == 1:
        axes = [axes]
    for ax, record in zip(axes, records):
        gate = record['gate'][0].numpy()
        alpha = record['alpha'].numpy()
        score = np.abs(alpha) * gate
        top_indices = np.argsort(score)[-topk:][::-1]
        ax.bar(np.arange(len(top_indices)), score[top_indices], color='#59a14f')
        ax.set_title(f'Stage {record["stage"]} top-{topk} |alpha| * gate channels')
        ax.set_xlabel('ranked channel')
        ax.set_ylabel('score')
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def write_csv(records, output_path, image_name, pred_class, pred_prob):
    fieldnames = [
        'image', 'pred_class', 'pred_prob', 'stage', 'channels',
        'gate_mean', 'gate_std', 'gate_min', 'gate_max',
        'alpha_abs_mean', 'global_delta_abs_mean', 'gated_delta_abs_mean',
    ]
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            gate = record['gate']
            writer.writerow({
                'image': image_name,
                'pred_class': pred_class,
                'pred_prob': f'{pred_prob:.6f}',
                'stage': record['stage'],
                'channels': gate.shape[1],
                'gate_mean': f'{gate.mean().item():.6f}',
                'gate_std': f'{gate.std().item():.6f}',
                'gate_min': f'{gate.min().item():.6f}',
                'gate_max': f'{gate.max().item():.6f}',
                'alpha_abs_mean': f'{record["alpha"].abs().mean().item():.8f}',
                'global_delta_abs_mean': f'{record["global_delta"].abs().mean().item():.8f}',
                'gated_delta_abs_mean': f'{record["gated_delta"].abs().mean().item():.8f}',
            })


def save_npz(records, output_path):
    payload = {}
    for record in records:
        stage = record['stage']
        payload[f'stage{stage}_gate'] = record['gate'].numpy()
        payload[f'stage{stage}_alpha'] = record['alpha'].numpy()
        payload[f'stage{stage}_gated_delta_abs_map'] = (
            record['gated_delta'].abs().mean(dim=1).numpy())
        payload[f'stage{stage}_global_delta_abs_map'] = (
            record['global_delta'].abs().mean(dim=1).numpy())
    np.savez_compressed(output_path, **payload)


def collect_images(args):
    if args.image_dir:
        paths = [
            p for p in sorted(Path(args.image_dir).rglob('*'))
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if args.max_images > 0:
            paths = paths[:args.max_images]
        return paths
    return [Path(args.image)]


def analyze_one(model, records, image_path, args):
    records.clear()
    tensor, display = load_image(image_path, args.img_size)
    tensor = tensor.to(args.device)
    with torch.no_grad():
        logits = model(tensor)
        prob = logits.softmax(dim=1)
        pred_prob, pred_class = prob.max(dim=1)

    pred_class = int(pred_class.item())
    pred_prob = float(pred_prob.item())
    if not records:
        raise RuntimeError('No gate records were captured. Check whether the model uses Galerkin and gate.')

    stem = image_path.stem
    out_dir = Path(args.output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    save_gate_bar(records, out_dir / 'gate_stage_bar.png', args.dpi)
    save_stage_maps(records, display, out_dir / 'gated_global_contribution_maps.png',
                    args.img_size, args.alpha, args.dpi)
    save_channel_maps(records, out_dir / 'gate_top_channels.png', args.dpi)
    write_csv(records, out_dir / 'gate_stats.csv', image_path.name, pred_class, pred_prob)
    save_npz(records, out_dir / 'gate_records.npz')
    print(f'{image_path.name}: class={pred_class}, prob={pred_prob:.4f}, saved={out_dir}')


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    records = install_gate_recorder(model)
    image_paths = collect_images(args)
    print(f'images: {len(image_paths)}')
    for image_path in image_paths:
        analyze_one(model, records, image_path, args)


if __name__ == '__main__':
    main()
