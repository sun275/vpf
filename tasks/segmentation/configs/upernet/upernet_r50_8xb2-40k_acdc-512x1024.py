_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/acdc.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]

data_root = '/data1/sunmy/ACDC/mmseg_format'
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=19),
    auxiliary_head=dict(num_classes=19),
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

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=10000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000))
