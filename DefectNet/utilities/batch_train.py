# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval
import mmcv
import copy
import json
import os.path as osp
from train import main as train_main
from defect_test import main as test_main
from infer import main as infer_main

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


def hint(wav_file='./wav/qq.wav', n=5):
    import pygame
    for i in range(n):
        pygame.mixer.init()
        pygame.mixer.music.load(wav_file)
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()


def batch_train(cfgs, sleep_time=0):
    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))

        # train
        train_params = dict(config=cfg)
        train_main(**train_params)
        print('{} train successfully!'.format(cfg_name))
        hint()
        time.sleep(sleep_time)
    # infer
    # infer_params = dict(
    #     config=cfg,
    #     resume_from=osp.join(cfg.work_dir, 'epoch_12.pth'),
    #     img_dir=osp.join(cfg.data_root, 'test'),
    #     work_dir=cfg.work_dir,
    #     submit_out=osp.join(cfg.work_dir, '{}_submit_epoch_12.json'.format(cfg_name)),
    #     have_bg=True,
    # )
    # infer_main(**infer_params)
    # print('{} infer successfully!'.format(cfg_name))
    # hint()


def fixed_defect_finding_weight_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]

    # watch train effects using different base cfg
    ratios = [0.1 * i for i in range(0, 21, 1)]
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
        cfg.model['find_weight'] = n
        cfg.cfg_name = 'fixed_defect_finding_weight'
        cfg.uid = None
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name, 'fixed_defect_finding_weight={:.1f}'.format(n))

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None
        cfgs.append(cfg)
    batch_train(cfgs, sleep_time=60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'fixed_defect_finding_weight.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')
    batch_test(cfgs, save_path, 60 * 2, mode='test')


def main():
    fixed_defect_finding_weight_train()


if __name__ == '__main__':
    main()
