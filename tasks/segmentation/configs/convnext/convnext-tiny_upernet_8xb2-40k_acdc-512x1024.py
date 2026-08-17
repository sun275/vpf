_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/datasets/acdc.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]

data_root = '/data1/sunmy/ACDC/mmseg_format'
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        arch='tiny',
        drop_path_rate=0.4,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),
    ),
    decode_head=dict(num_classes=19, in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(num_classes=19, in_channels=384),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 682)),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(data_root=data_root),
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    ),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=10000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000)
)
