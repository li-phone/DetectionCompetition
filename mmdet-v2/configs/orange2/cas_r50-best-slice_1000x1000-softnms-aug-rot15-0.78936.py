# classes
CLASSES = (
    'bug', 'fruit_bug'
)

# fp16 settings
fp16 = dict(loss_scale=512.)

# SyncBN or BN
norm_cfg = dict(type='BN', requires_grad=True)
# norm_cfg = dict(type='SyncBN', requires_grad=True)

# model settings
num_classes = len(CLASSES)
model = dict(
    type='CascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=norm_cfg,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            clip_border=True,  # 不允许超出图像大小
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        # norm_cfg=norm_cfg,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            # norm_cfg=norm_cfg,
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # type='DoubleConvFCBBoxHead',
                norm_cfg=norm_cfg,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                    clip_border=True,  # 不允许超出图像大小
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                # type='DoubleConvFCBBoxHead',
                norm_cfg=norm_cfg,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                    clip_border=True,  # 不允许超出图像大小
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                # type='DoubleConvFCBBoxHead',
                norm_cfg=norm_cfg,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                    clip_border=True,  # 不允许超出图像大小
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            gpu_assign_thr=128,
            iou_calculator=dict(
                type='BboxOverlaps2D',
                scale=512.,
                dtype='fp16',
            ),
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            # 增加采样 * 2
            # num=512,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=2000,
        max_per_img=2000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    # rpn_proposal=dict(
    #     nms_across_levels=False,
    #     nms_pre=2000,
    #     nms_post=2000,
    #     max_num=2000,
    #     nms_thr=0.7,
    #     min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                gpu_assign_thr=128,
                iou_calculator=dict(
                    type='BboxOverlaps2D',
                    scale=512.,
                    dtype='fp16',
                ),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                # 增加采样 * 2
                # num=1024,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                gpu_assign_thr=128,
                iou_calculator=dict(
                    type='BboxOverlaps2D',
                    scale=512.,
                    dtype='fp16',
                ),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                # 增加采样 * 2
                # num=1024,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                match_low_quality=False,
                gpu_assign_thr=128,
                iou_calculator=dict(
                    type='BboxOverlaps2D',
                    scale=512.,
                    dtype='fp16',
                ),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                # 增加采样 * 2
                # num=1024,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ])
test_cfg = dict(
    # rpn=dict(
    #     nms_across_levels=False,
    #     nms_pre=1000,
    #     nms_post=1000,
    #     max_num=1000,
    #     nms_thr=0.7,
    #     min_bbox_size=0),
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.0001,
        # nms=dict(type='nms', iou_threshold=0.5),
        nms=dict(type='soft_nms', iou_thr=0.5),
        max_per_img=2000))

dataset_type = 'CocoDataset'
data_root = 'data/orange2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='HorizontalFlip',
        p=0.2),
    dict(
        type='VerticalFlip',
        p=0.2),
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=180,
    #     interpolation=1,
    #     p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.3),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=10,
    #             g_shift_limit=10,
    #             b_shift_limit=10,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0),
    #             dict(type='FancyPCA', alpha=0.1, always_apply=False, p=1.0), #trick
    #     ],
    #     p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    # dict(type='RandomContrast', limit=0.2, always_apply=False, p=0.3),
    dict(type='RandomRotate90', always_apply=False, p=0.5),  # 随机旋转
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0),
    #         dict(type='MotionBlur', blur_limit=6, always_apply=False, p=1.0)#trick
    #     ],
    #     p=0.1),
]

policies1 = [
    [dict(type='RandomFlip', flip_ratio=0.0, direction='horizontal')],
    [dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal')],
    [dict(type='RandomFlip', flip_ratio=0.5, direction='vertical')],
    [dict(type='RandomFlip', flip_ratio=0.5, direction='diagonal')],
]
policies2 = [
    [dict(type='Rotate', level=10, prob=0.5, max_rotate_angle=0)],
    [dict(type='Rotate', level=10, prob=0.5, max_rotate_angle=90)],
    [dict(type='Rotate', level=10, prob=0.5, max_rotate_angle=180)],
    [dict(type='Rotate', level=10, prob=0.5, max_rotate_angle=270)],
]
policies3 = [
    [dict(type='Rotate', level=10, prob=0.5, max_rotate_angle=0)],
    [dict(type='Rotate', level=10, prob=0.5, max_rotate_angle=15)],
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1333, 800), ratio_range=(0.8, 1.2), keep_ratio=True),
    # dict(type='Resize', img_scale=(1000, 1000), ratio_range=(0.8, 1.2), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment', policies=policies1),
    dict(type='AutoAugment', policies=policies2),
    dict(type='AutoAugment', policies=policies3),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),
    # dict(type='Translate', level=8),
    # dict(type='ColorTransform', level=8),
    # dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    # dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    # dict(type='CutOut', n_holes=2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(600, 600), (800, 800), (1000, 1000)],
        # img_scale=[(800, 800), (1000, 1000), (1200, 1200), ],
        img_scale=[(1066, 640), (1333, 800), (1600, 960)],
        # img_scale=(1333, 800),
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
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/slice_1000x1000_train.json',
        img_prefix=data_root + 'slice_1000x1000',
        classes=CLASSES,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/slice_1000x1000_train.json',
        img_prefix=data_root + 'slice_1000x1000',
        classes=CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/slice_1000x1000_train.json',
        img_prefix=data_root + 'slice_1000x1000',
        classes=CLASSES,
        pipeline=test_pipeline),
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
    step=[8, 11],
    # step=[16, 19],
    # step=[16, 22]
)
total_epochs = 12

checkpoint_config = dict(interval=6)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/cas_r50-best-slice_1000x1000-auto_aug/latest.pth'
# resume_from = 'work_dirs/cas_r50-best/latest.pth'
resume_from = None
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]
