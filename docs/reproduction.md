# Reproduction

Each task directory contains its own configs and entry points while importing
the same model from `src/vpf`.

Downstream configs intentionally set `model.backbone.pretrained=None`. Supply a
classification checkpoint explicitly:

```bash
--cfg-options model.backbone.pretrained=/absolute/path/to/checkpoint.pth
```

Training outputs should be written below `outputs/`. Do not commit datasets,
checkpoints, or raw logs; publish checkpoints through a release or model hub.
