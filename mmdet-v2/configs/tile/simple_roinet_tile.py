# model settings
model = dict(
    type='ROIRCNN',
    pretrained='torchvision://resnet50',
    roi_bone=dict(type='SimpleROINet'),
)
# model training and testing settings
train_cfg = dict()
test_cfg = dict()
dataset_type = 'TileDataset'
data_root = 'data/tile/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/cut_800x800/cut_800x800_train.json',
        img_prefix=data_root + 'trainval/cut_800x800',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/cut_800x800/cut_800x800_val.json',
        img_prefix=data_root + 'trainval/cut_800x800',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/backup/instance_XS_train.json',
        img_prefix=data_root + 'raw/tile_round1_train_20201231/train_imgs',
        pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/cut_800x800/cut_800x800_val.json',
    #     img_prefix=data_root + 'trainval/cut_800x800',
    #     pipeline=test_pipeline),
)
evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.02 / 4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data/liphone/detcomp/mmdet-v2/tile/baseline_cut_1000x1000_2/epoch_12.pth'
resume_from = None
work_dir = './work_dirs/tile/simple_roinet_tile'
workflow = [('train', 1), ('val', 1)]
