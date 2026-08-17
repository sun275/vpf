_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

data_root = 'data/coco/'

model = dict(
    backbone=dict(
        _delete_=True,
        type='MMDET_VPF',
        drop_path_rate=0.1,
        post_norm=False,
        depths=(2, 2, 5, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        img_size=224,
        ablation='full',
        pretrained=None,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    batch_size=2,
    dataset=dict(data_root=data_root),
)
test_dataloader = dict(
    batch_size=2,
    dataset=dict(data_root=data_root),
)
val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
)
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
)
