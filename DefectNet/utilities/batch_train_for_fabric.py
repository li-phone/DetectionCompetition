from train_with_tricks import BatchTrain
import numpy as np


def main():
    aquatic_train = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/aquatic.py', data_mode='val')
    aquatic_train.compete_train()

    # garbage_train = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/garbage.py', data_mode='val')
    # garbage_train.joint_train()

    fabric_train = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/fabric.py', data_mode='test')
    fabric_train.anchor_cluster_train(anchor_ratios=[0.12, 1.0, 4.43])
    fabric_train.anchor_cluster_train(anchor_ratios=[0.04, 0.28, 1.0, 4.43, 8.77])
    fabric_train.anchor_cluster_train(anchor_ratios=[0.04, 0.14, 0.32, 0.69, 1.0, 4.77, 8.84])
    fabric_train.anchor_cluster_train(anchor_ratios=[0.04, 0.14, 0.27, 0.32, 0.69, 1.0, 3.0, 5.71, 10.22])
    # batrian.baseline_train()
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)], ratio_range=[0.5, 1.5])
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)], multiscale_mode='value')
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)])
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)])
    # batrian.multi_scale_train(img_scale=[(2446, 1000)])


if __name__ == '__main__':
    main()
