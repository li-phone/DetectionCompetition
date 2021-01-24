_base_ = './cascade_rcnn_r50_fpn_1x_cut_800x800.py'
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
