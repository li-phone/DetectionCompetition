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
        # "../configs/orange2/cas_r101-best_base-resize_0.5_1.0-iou_0.5_score_0.01.py",
        # "../configs/orange2/cas_r101-best_base-resize_0.5_1.0-iou_0.5_score_0.001.py",
        "../configs/orange2/cas_r101-best_base-resize_0.5_1.0-iou_0.8_score_0.3.py",
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
