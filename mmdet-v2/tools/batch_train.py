import glob
import os
import os.path as osp
from tqdm import tqdm
from mmcv import Config, DictAction
from train import main as train_main
from finetune_data import main as finetune_main


def finetune_train(cfg_path):
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = 'work_dirs/cas_r50-best/latest.pth'
    total_epochs = cfg.total_epochs
    for epoch in range(1, total_epochs + 1):
        cfg.total_epochs = epoch
        save_name = cfg.data_root + f'annotations/instance-train-finetune-{epoch}.json'
        # 先调整训练数据集
        finetune_main(cfg_path, cfg.load_from, cfg.data.train.ann_file, save_name)
        cfg.data.train.ann_file = save_name
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(cfg_path))[0])
        cfg.work_dirs = osp.join('./work_dirs',
                                 osp.splitext(osp.basename(cfg_path))[0])
        train_main(cfg)
        cfg.load_from = cfg.work_dirs + '/latest.pth'


def main():
    # configs=glob.glob("../configs/goods/*.py")
    configs = [
        # "../configs/orange/detectoRS-cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-finetune_data_iou_0.8_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-finetune_data_iou_0.6_score_0.8_iter_1.py",
        # "../configs/orange/cas_r101v1d-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/detectoRS-cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-lr_0.0125.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-pafpn.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-roi_14.py",
        # "../configs/orange/cas_r101-best.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-pseudo_label.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-scale_4.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-ratio_7.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-giou.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-fp32.py",
        # "../configs/orange/detectoRS-cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-mask.py",
        # "../configs/orange/cas_r50-iou_0.7_score_0.8-predict-iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-iou_0.7_score_0.8-predict-iou_0.8_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-iou_0.7_score_0.8-predict-iou_0.9_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-albu.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-mosaic.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-dh-rpn-3-DIouLoss.py",
        # "../configs/orange/cas_r50-rpn_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-rpn_iou_0.8_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-rpn_iou_0.9_score_0.8_iter_1.py",
        # "../configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py",
        # "../configs/orange2/cas_r50-best-slice_800x800.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000_3000x3000.py",
        # "../configs/orange2/cas_r50-best-slice_1333x800.py",
        # "../configs/orange2/cas_r50-best-slice_1333x1333.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000-img_scale_1666x1000.py",
        # "../configs/orange2/cas_r50-best-slice_1333x1333-img_scale_2221x1333.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000-softnms-auto_aug-rot15.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000-softnms-auto_aug-slice_800x800.py",
        # "../configs/orange2/cas_r101-best-slice_1000x1000-softnms-aug-800x800.py",
        # "../configs/orange2/cas_x101-best-slice_1000x1000-softnms-aug-800x800.py",
        # "../configs/orange2/cas_r101-best_base-1000x1000_ovlap_0.5.py",
        # "../configs/orange2/cas_r101-best_base-anchor_ratio.py",
        # "../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5.py",
        "../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0_1.5.py",
        # "../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0.py",
    ]
    configs.sort()
    for cfg_path in tqdm(configs):
        # train_main(cfg_path)
        try:
            train_main(cfg_path)
        except Exception as e:
            print(cfg_path, 'train error!', e)
    print('train all config ok!')


if __name__ == '__main__':
    main()
