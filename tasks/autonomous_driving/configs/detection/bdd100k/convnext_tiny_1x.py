_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

data_root = 'data/bdd100k/'
classes = (
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle', 'traffic light', 'traffic sign'
)
metainfo = dict(classes=classes)
backend_args = None

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
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
        ann_file='bdd100k_det_10cls_coco/annotations/train_10cls_coco.json',
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
        ann_file='bdd100k_det_10cls_coco/annotations/val_10cls_coco.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'bdd100k_det_10cls_coco/annotations/val_10cls_coco.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1,
    )
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6,
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)

auto_scale_lr = dict(enable=False, base_batch_size=16)
