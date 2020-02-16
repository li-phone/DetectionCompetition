# model settings
model = dict(
    type='FirstModel',
    num_stages=3,
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        num_classes=2,
        loss_cls=dict(
            type='nn.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
)
# model training and testing settings
train_cfg = dict()
test_cfg = dict()
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/liphone/undone-work/data/detection/alcohol'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(224, 224),),
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
        img_scale=(1333, 800),
        flip=False,
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
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/instance_train_alcohol.json',
        img_prefix=data_root + '/trainval/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/instance_train_alcohol.json',
        img_prefix=data_root + '/trainval/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/instance_test_alcohol.json',
        img_prefix=data_root + '/trainval/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dataset_name = 'alcohol'
total_epochs = 12 * 4
dist_params = dict(backend='nccl')
log_level = 'INFO'
uid = None
cfg_name = ''
work_dir = '../work_dirs/' + dataset_name + '/cascade_rcnn_r50_fpn_1x/' + cfg_name
resume_from = None
load_from = None
workflow = [('train', 1)]
