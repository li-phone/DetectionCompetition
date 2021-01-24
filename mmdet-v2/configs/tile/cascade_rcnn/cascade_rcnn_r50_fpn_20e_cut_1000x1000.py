_base_ = './cascade_rcnn_r50_fpn_1x_cut_1000x1000.py'
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
