# Dataset layout

The default configs expect datasets below the repository-level `data/` folder:

```text
data/
├── imagenet/
│   ├── train/<class>/*.JPEG
│   └── val/<class>/*.JPEG
├── coco/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
├── ade20k/
│   ├── images/
│   └── annotations/
├── bdd100k/
│   └── bdd100k_det_10cls_coco/
├── cityscapes/
└── acdc/
    └── mmseg_format/
```

The `data/` directory is ignored by Git. Symlinks are suitable. For a different
location, override `data_root` fields with `--cfg-options` or edit a local copy
of the relevant config.
