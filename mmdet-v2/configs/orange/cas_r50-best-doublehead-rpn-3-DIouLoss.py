# classes
CLASSES = (
    'HLB', 'health', 'ill'
)

# fp16 settings
fp16 = dict(loss_scale=512.)

# SyncBN or BN
norm_cfg = dict(type='BN', requires_grad=True)
# norm_cfg = dict(type='SyncBN', requires_grad=True)

# model settings
num_classes = len(CLASSES)
rpn_weight = 0.7
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
            type='CascadeRPNHead',
            num_stages=4,
            stages=[
                dict(
                    type='StageCascadeRPNHead',
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[8],
                        ratios=[1.0],
                        strides=[4, 8, 16, 32, 64]),
                    adapt_cfg=dict(type='dilation', dilation=3),
                    bridged_feature=True,
                    sampling=False,
                    with_cls=False,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.1, 0.1, 0.5, 0.5)),
                    loss_bbox=dict(
                        type='DIoULoss',                        #todo 原始为DIoULoss
                        loss_weight=10.0 * rpn_weight)),
                dict(
                    type='StageCascadeRPNHead',
                    in_channels=256,
                    feat_channels=256,
                    adapt_cfg=dict(type='offset'),
                    bridged_feature=False,
                    sampling=True,
                    with_cls=True,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.05, 0.05, 0.1, 0.1)),
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0 * rpn_weight),
                    loss_bbox=dict(
                        type='DIoULoss', #todo linear=True,
                        loss_weight=10.0 * rpn_weight)),
                dict(
                    type='StageCascadeRPNHead',
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[8],
                        ratios=[1.0],
                        strides=[4, 8, 16, 32, 64]),
                    adapt_cfg=dict(type='dilation', dilation=1),  #todo 3-->1
                    bridged_feature=True,
                    sampling=False,
                    with_cls=False,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.05, 0.05, 0.1, 0.1)),
                    loss_bbox=dict(
                        type='DIoULoss',  # todo 原始为DIoULoss
                        loss_weight=10.0 * rpn_weight)),
                dict(
                    type='StageCascadeRPNHead',
                    in_channels=256,
                    feat_channels=256,
                    adapt_cfg=dict(type='offset'),
                    bridged_feature=False,
                    sampling=True,
                    with_cls=True,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.033, 0.033, 0.067, 0.067)),
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0 * rpn_weight),
                    loss_bbox=dict(
                        type='DIoULoss', #todo linear=True,
                        loss_weight=10.0 * rpn_weight))
            ]),
    roi_head=dict(
        type='CascadeRoIHead',
        # norm_cfg=norm_cfg,
        num_stages=3,
        reg_roi_scale_factor=1,  #todo 原始1.3
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            # norm_cfg=norm_cfg,
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=4,
                num_fcs=2,
                norm_cfg=norm_cfg,
                in_channels=256,
                conv_out_channels=1024,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                    clip_border=True,  #fixme  不允许超出图像大小
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                #loss_bbox=dict(type='SmoothL1Loss', beta=1.0,loss_weight=1.0)),
                loss_bbox = dict(
                    type='BalancedL1Loss',
                    alpha=0.5,
                    gamma=1.5,
                    beta=1.0,
                    loss_weight=1.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=4,
                num_fcs=2,
                norm_cfg=norm_cfg,
                in_channels=256,
                conv_out_channels=1024,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                    clip_border=True,  #fixme 不允许超出图像大小
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                #loss_bbox=dict(type='SmoothL1Loss', beta=1.0,loss_weight=1.0)),
                loss_bbox=dict(
                    type='BalancedL1Loss',
                    alpha=0.5,
                    gamma=1.5,
                    beta=1.0,
                    loss_weight=1.0)),
            dict(
                type='DoubleConvFCBBoxHead',
                num_convs=4,
                num_fcs=2,
                norm_cfg=norm_cfg,
                in_channels=256,
                conv_out_channels=1024,
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
                #loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
                loss_bbox=dict(
                    type='BalancedL1Loss',
                    alpha=0.5,
                    gamma=1.5,
                    beta=1.0,
                    loss_weight=1.0))
        ]),

    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='RegionAssigner',
                    center_ratio=0.2,
                    ignore_ratio=0.5),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    gpu_assign_thr=64,
                    iou_calculator=dict(
                        type='BboxOverlaps2D',
                        scale=512.,
                        dtype='fp16',
                    ),
                ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='RegionAssigner',
                    center_ratio=0.2,
                    ignore_ratio=0.5),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    gpu_assign_thr=64,
                    iou_calculator=dict(
                        type='BboxOverlaps2D',
                        scale=512.,
                        dtype='fp16',
                    ),
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False)
        ],
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    gpu_assign_thr=64,
                    iou_calculator=dict(
                        type='BboxOverlaps2D',
                        scale=512.,
                        dtype='fp16',
                    ),
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
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
                    gpu_assign_thr=64,
                    iou_calculator=dict(
                        type='BboxOverlaps2D',
                        scale=512.,
                        dtype='fp16',
                    ),
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
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
                    gpu_assign_thr=64,
                    iou_calculator=dict(
                        type='BboxOverlaps2D',
                        scale=512.,
                        dtype='fp16',
                    ),
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,nms=dict(type='nms', iou_threshold=0.5),max_per_img=100)
    )
)

dataset_type = 'CocoDataset'
data_root = 'data/orange/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1333, 800), ratio_range=(0.8, 1.2), keep_ratio=True),
    # dict(type='Resize', img_scale=(1000, 1000), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
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
        ann_file=data_root + 'annotations/instance-train-best-iou_0.7_score_0.8_iter_1.json',
        img_prefix=data_root + 'train/images',
        classes=CLASSES,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instance-train.json',
        img_prefix=data_root + 'train/images',
        classes=CLASSES,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instance-train.json',
        img_prefix=data_root + 'train/images',
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
load_from = 'work_dirs/cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1/latest.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
