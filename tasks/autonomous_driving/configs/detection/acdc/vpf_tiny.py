_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

data_root = 'data/acdc/'
classes = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck')
metainfo = dict(classes=classes)

backend_args = None

model = dict(
    backbone=dict(
        _delete_=True,
        type='MMDET_VPF',
        drop_path_rate=0.2,
        post_norm=False,
        depths=(2, 3, 4, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        img_size=224,
        ablation='full',
        pretrained=None,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(num_classes=6),
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoiceResize',
        scales=[(1280, 640), (1280, 672), (1280, 704), (1280, 736),
                (1280, 768), (1280, 800)],
        keep_ratio=True),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='acdc_6cls_cocojson/annotations/train_6cls_coco.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='acdc_6cls_cocojson/annotations/val_6cls_coco.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'acdc_6cls_cocojson/annotations/val_6cls_coco.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)

max_epochs = 36
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1,
    )
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
