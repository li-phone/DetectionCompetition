_base_ = './cascade_rcnn_r50_fpn_20e_cut_1000x1000.py'
model = dict(
    type='CascadeRCNN',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20

work_dir = './work_dirs/tile/cascade_rcnn_x101_64x4d_fpn_20e_cut_1000x1000'
load_from = '/data/liphone/detcomp/mmdet-v2/tile/baseline_cut_1000x1000_2/epoch_12.pth'
