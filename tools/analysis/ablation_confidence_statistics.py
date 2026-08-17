#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


THIS_DIR = Path(__file__).resolve().parent
CLASSIFICATION_ROOT = THIS_DIR.parent
REPO_ROOT = CLASSIFICATION_ROOT.parent
sys.path.insert(0, str(CLASSIFICATION_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'visual'))

from vpf.models import vpf  # noqa: E402
from vpf_true_ablation import apply_true_no_dno  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ABLATIONS = ('no_galerkin', 'no_gate', 'no_dno')


def parse_int_tuple(text):
    return tuple(int(v.strip()) for v in text.split(',') if v.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Forward-only confidence statistics for VPF ablation variants.')
    parser.add_argument(
        '--data-root',
        default='/data1/sunmy/Imagenet1k/data/val',
        help='ImageFolder validation root.')
    parser.add_argument(
        '--checkpoint',
        default='/data/sunmy/vpf/vpf_aban_2342/vpf_tiny/default/ckpt_epoch_ema_best.pth',
        help='VPF checkpoint path.')
    parser.add_argument(
        '--output-dir',
        default=str(CLASSIFICATION_ROOT / 'analysis' / 'gate_outputs_test' / 'ablation_confidence'),
        help='Output directory.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--depths', default='2,3,4,2')
    parser.add_argument('--dims', type=int, default=96)
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--max-images', type=int, default=0, help='0 means all images.')
    parser.add_argument('--amp', action='store_true', help='Use CUDA autocast for faster inference.')
    parser.add_argument(
        '--model-no-dno',
        action='store_true',
        help='Use model ablation=no_dno instead of true DNO removal. Default matches Grad-CAM visualization.')
    parser.add_argument(
        '--save-per-image',
        action='store_true',
        help='Save per-image comparison csv. Usually not needed.')
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
    model_ablation = ablation
    true_no_dno = ablation == 'no_dno' and not args.model_no_dno
    if true_no_dno:
        model_ablation = 'full'

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
        ablation=model_ablation,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if true_no_dno:
        apply_true_no_dno(model)
    print(f'[{ablation}] missing={len(incompatible.missing_keys)}, '
          f'unexpected={len(incompatible.unexpected_keys)}'
          f'{" true_no_dno" if true_no_dno else ""}')
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


def forward_logits(model, images, args):
    use_amp = args.amp and args.device.startswith('cuda')
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            return model(images)


def topk_correct(logits, targets, k=5):
    pred = logits.topk(k, dim=1).indices
    return pred.eq(targets[:, None]).any(dim=1)


def run_full_pass(loader, model, args, num_images):
    labels = np.empty(num_images, dtype=np.int64)
    full_pred = np.empty(num_images, dtype=np.int64)
    full_correct = np.empty(num_images, dtype=np.bool_)
    full_top5_correct = np.empty(num_images, dtype=np.bool_)
    full_top1_prob = np.empty(num_images, dtype=np.float32)
    full_top1_logit = np.empty(num_images, dtype=np.float32)
    full_gt_prob = np.empty(num_images, dtype=np.float32)
    full_gt_logit = np.empty(num_images, dtype=np.float32)

    offset = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        logits = forward_logits(model, images, args).float()
        probs = logits.softmax(dim=1)
        top_prob, pred = probs.max(dim=1)
        top_logit = logits.gather(1, pred[:, None]).squeeze(1)
        gt_prob = probs.gather(1, targets[:, None]).squeeze(1)
        gt_logit = logits.gather(1, targets[:, None]).squeeze(1)
        bs = images.shape[0]
        sl = slice(offset, offset + bs)
        labels[sl] = targets.cpu().numpy()
        full_pred[sl] = pred.cpu().numpy()
        full_correct[sl] = pred.eq(targets).cpu().numpy()
        full_top5_correct[sl] = topk_correct(logits, targets, 5).cpu().numpy()
        full_top1_prob[sl] = top_prob.cpu().numpy()
        full_top1_logit[sl] = top_logit.cpu().numpy()
        full_gt_prob[sl] = gt_prob.cpu().numpy()
        full_gt_logit[sl] = gt_logit.cpu().numpy()
        offset += bs
        if batch_idx % 20 == 0:
            print(f'[full] processed {offset}/{num_images}')

    return {
        'labels': labels,
        'full_pred': full_pred,
        'full_correct': full_correct,
        'full_top5_correct': full_top5_correct,
        'full_top1_prob': full_top1_prob,
        'full_top1_logit': full_top1_logit,
        'full_gt_prob': full_gt_prob,
        'full_gt_logit': full_gt_logit,
    }


def run_ablation_pass(loader, model, args, full):
    num_images = len(full['labels'])
    pred = np.empty(num_images, dtype=np.int64)
    correct = np.empty(num_images, dtype=np.bool_)
    top5_correct_arr = np.empty(num_images, dtype=np.bool_)
    top1_prob = np.empty(num_images, dtype=np.float32)
    top1_logit = np.empty(num_images, dtype=np.float32)
    full_target_prob = np.empty(num_images, dtype=np.float32)
    full_target_logit = np.empty(num_images, dtype=np.float32)
    gt_prob = np.empty(num_images, dtype=np.float32)
    gt_logit = np.empty(num_images, dtype=np.float32)

    offset = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        bs = images.shape[0]
        sl = slice(offset, offset + bs)
        fixed_full_target = torch.from_numpy(full['full_pred'][sl]).to(args.device)

        logits = forward_logits(model, images, args).float()
        probs = logits.softmax(dim=1)
        batch_top_prob, batch_pred = probs.max(dim=1)
        batch_top_logit = logits.gather(1, batch_pred[:, None]).squeeze(1)
        batch_full_target_prob = probs.gather(1, fixed_full_target[:, None]).squeeze(1)
        batch_full_target_logit = logits.gather(1, fixed_full_target[:, None]).squeeze(1)
        batch_gt_prob = probs.gather(1, targets[:, None]).squeeze(1)
        batch_gt_logit = logits.gather(1, targets[:, None]).squeeze(1)

        pred[sl] = batch_pred.cpu().numpy()
        correct[sl] = batch_pred.eq(targets).cpu().numpy()
        top5_correct_arr[sl] = topk_correct(logits, targets, 5).cpu().numpy()
        top1_prob[sl] = batch_top_prob.cpu().numpy()
        top1_logit[sl] = batch_top_logit.cpu().numpy()
        full_target_prob[sl] = batch_full_target_prob.cpu().numpy()
        full_target_logit[sl] = batch_full_target_logit.cpu().numpy()
        gt_prob[sl] = batch_gt_prob.cpu().numpy()
        gt_logit[sl] = batch_gt_logit.cpu().numpy()
        offset += bs
        if batch_idx % 20 == 0:
            print(f'[ablation] processed {offset}/{num_images}')

    return {
        'pred': pred,
        'correct': correct,
        'top5_correct': top5_correct_arr,
        'top1_prob': top1_prob,
        'top1_logit': top1_logit,
        'full_target_prob': full_target_prob,
        'full_target_logit': full_target_logit,
        'gt_prob': gt_prob,
        'gt_logit': gt_logit,
    }


def summarize_ablation(name, full, ab):
    n = len(full['labels'])
    full_correct = full['full_correct']
    ab_correct = ab['correct']

    fixed_prob_higher = ab['full_target_prob'] > full['full_top1_prob']
    fixed_logit_higher = ab['full_target_logit'] > full['full_top1_logit']
    own_prob_higher = ab['top1_prob'] > full['full_top1_prob']
    same_pred = ab['pred'] == full['full_pred']

    return {
        'model': name,
        'num_images': int(n),
        'top1_acc': float(ab_correct.mean() * 100.0),
        'top5_acc': float(ab['top5_correct'].mean() * 100.0),
        'same_pred_as_full': float(same_pred.mean() * 100.0),
        'full_correct_ablation_wrong': float((full_correct & ~ab_correct).mean() * 100.0),
        'ablation_correct_full_wrong': float((ab_correct & ~full_correct).mean() * 100.0),
        'both_correct': float((full_correct & ab_correct).mean() * 100.0),
        'both_wrong': float((~full_correct & ~ab_correct).mean() * 100.0),
        'fixed_full_prob_higher_than_full': float(fixed_prob_higher.mean() * 100.0),
        'fixed_full_logit_higher_than_full': float(fixed_logit_higher.mean() * 100.0),
        'own_top1_prob_higher_than_full_top1': float(own_prob_higher.mean() * 100.0),
        'fixed_full_prob_higher_but_ablation_wrong': float((fixed_prob_higher & ~ab_correct).mean() * 100.0),
        'own_top1_prob_higher_but_ablation_wrong': float((own_prob_higher & ~ab_correct).mean() * 100.0),
        'mean_fixed_full_prob_delta': float((ab['full_target_prob'] - full['full_top1_prob']).mean()),
        'mean_fixed_full_logit_delta': float((ab['full_target_logit'] - full['full_top1_logit']).mean()),
        'mean_gt_prob_delta': float((ab['gt_prob'] - full['full_gt_prob']).mean()),
        'mean_gt_logit_delta': float((ab['gt_logit'] - full['full_gt_logit']).mean()),
    }


def save_summary(rows, full_summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'ablation_confidence_summary.csv'
    fieldnames = list(rows[0].keys())
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'full': full_summary,
        'ablations': rows,
    }
    (output_dir / 'ablation_confidence_summary.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')


def save_per_image(output_dir, full, ablation_outputs):
    path = output_dir / 'ablation_confidence_per_image.csv'
    fieldnames = [
        'index', 'label', 'full_pred', 'full_correct', 'full_top1_prob',
        'ablation', 'ab_pred', 'ab_correct', 'same_pred_as_full',
        'ab_top1_prob', 'ab_full_target_prob', 'ab_full_target_prob_minus_full',
        'ab_full_target_logit_minus_full',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, ab in ablation_outputs.items():
            for i in range(len(full['labels'])):
                writer.writerow({
                    'index': i,
                    'label': int(full['labels'][i]),
                    'full_pred': int(full['full_pred'][i]),
                    'full_correct': bool(full['full_correct'][i]),
                    'full_top1_prob': float(full['full_top1_prob'][i]),
                    'ablation': name,
                    'ab_pred': int(ab['pred'][i]),
                    'ab_correct': bool(ab['correct'][i]),
                    'same_pred_as_full': bool(ab['pred'][i] == full['full_pred'][i]),
                    'ab_top1_prob': float(ab['top1_prob'][i]),
                    'ab_full_target_prob': float(ab['full_target_prob'][i]),
                    'ab_full_target_prob_minus_full': float(ab['full_target_prob'][i] - full['full_top1_prob'][i]),
                    'ab_full_target_logit_minus_full': float(ab['full_target_logit'][i] - full['full_top1_logit'][i]),
                })


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, num_images = make_loader(args)
    state_dict = strip_prefix(load_checkpoint(args.checkpoint))

    full_model = build_model(args, 'full', state_dict)
    full = run_full_pass(loader, full_model, args, num_images)
    release_model(full_model)

    full_summary = {
        'num_images': int(num_images),
        'top1_acc': float(full['full_correct'].mean() * 100.0),
        'top5_acc': float(full['full_top5_correct'].mean() * 100.0),
        'mean_top1_prob': float(full['full_top1_prob'].mean()),
        'mean_top1_logit': float(full['full_top1_logit'].mean()),
        'data_root': args.data_root,
        'checkpoint': args.checkpoint,
    }
    print('[full]', full_summary)

    rows = []
    ablation_outputs = {}
    for ablation in ABLATIONS:
        model = build_model(args, ablation, state_dict)
        ab = run_ablation_pass(loader, model, args, full)
        release_model(model)
        summary = summarize_ablation(ablation, full, ab)
        rows.append(summary)
        print(f'[{ablation}] {summary}')
        if args.save_per_image:
            ablation_outputs[ablation] = ab

    save_summary(rows, full_summary, output_dir)
    if args.save_per_image:
        save_per_image(output_dir, full, ablation_outputs)
    print(f'saved to {output_dir}')


if __name__ == '__main__':
    main()
