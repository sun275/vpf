_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

data_root = '/data1/sunmy/ADEChallengeData2016'
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        arch='tiny',
        drop_path_rate=0.4,
        init_cfg=None,
    ),
    decode_head=dict(num_classes=150, in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(num_classes=150, in_channels=384),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

train_dataloader = dict(
    batch_size=4,
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(data_root=data_root),
)
test_dataloader = val_dataloader
