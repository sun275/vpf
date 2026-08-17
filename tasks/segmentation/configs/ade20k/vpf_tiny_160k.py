_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]

data_root = 'data/ade20k/'
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='MMSEG_VPF',
        img_size=512,
        post_norm=False, 
        drop_path_rate=0.2,
        depths=(2, 3, 4, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        ablation='full',
        pretrained=None,
    ),
    decode_head=dict(num_classes=150, in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(num_classes=150, in_channels=384),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

train_dataloader = dict(
    batch_size=4,
    dataset=dict(data_root=data_root),
) # as gpus=8
val_dataloader = dict(
    batch_size=1,
    dataset=dict(data_root=data_root),
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
