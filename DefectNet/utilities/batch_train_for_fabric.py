from train_with_tricks import BatchTrain
import numpy as np


def main():
    # ann_file = '/home/liphone/undone-work/data/detection/aquatic/annotations/aquatic_train.json'
    # from coco_analyze import chg2coco
    # coco = chg2coco(ann_file)
    # img_sizes = [(r['width'], r['height']) for r in coco.dataset['images']]
    # img_sizes = set(img_sizes)
    resize_cfg = dict(
        img_scale=[(1920 * 2 / 3, 1080 * 2 / 3)],
        ratio_range=[0.5, 1.5],
        multiscale_mode='range',
        keep_ratio=True,
    )
    # resize_cfg = None
    garbage_train = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/garbage.py', data_mode='val')
    garbage_train.joint_train(resize_cfg)

    aquatic_train = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/aquatic.py', data_mode='val')
    aquatic_train.joint_train(resize_cfg)

    # batrian = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/fabric.py', data_mode='test')
    # batrian.baseline_train()
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)], ratio_range=[0.5, 1.5])
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)], multiscale_mode='value')
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)])
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)])
    # batrian.multi_scale_train(img_scale=[(2446, 1000)])


if __name__ == '__main__':
    main()
