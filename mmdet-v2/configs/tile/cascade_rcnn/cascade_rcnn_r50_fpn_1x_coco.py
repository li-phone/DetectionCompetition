_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/tile_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
total_epochs = 12
work_dir = './work_dirs/tile/baseline_cut_1000x1000_2'
load_from = '/data/liphone/detcomp/mmdet-v2/tile/baseline_cut_1000x1000/epoch_12.pth'
