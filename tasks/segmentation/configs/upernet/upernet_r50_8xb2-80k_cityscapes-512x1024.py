_base_ = './upernet_r50_4xb2-80k_cityscapes-512x1024.py'

data_root = '/data1/sunmy/cityscapes'

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

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=20000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000))
