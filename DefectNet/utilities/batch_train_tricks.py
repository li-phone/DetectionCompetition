# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval
import mmcv
import copy
import json
import numpy as np
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


def batch_infer(cfgs):
    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))
        infer_params = dict(
            config=cfg,
            resume_from=osp.join(cfg.work_dir, 'epoch_12.pth'),
            infer_object=cfg.data['test']['ann_file'],
            img_dir=cfg.data['test']['img_prefix'],
            work_dir=cfg.work_dir,
            submit_out=osp.join(cfg.work_dir, '{}_submit_epoch_{}.json'.format(cfg_name, 12)),
            have_bg=False,
        )
        infer_main(**infer_params)
        print('{} infer successfully!'.format(cfg_name))
        hint()


def batch_train(cfgs, sleep_time=0, detector=True):
    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))

        # train
        train_params = dict(config=cfg, detector=detector)
        train_main(**train_params)
        print('{} train successfully!'.format(cfg_name))
        hint()
        time.sleep(sleep_time)


def batch_fixed_defect_finding_weight_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]

    # watch train effects using different base cfg
    # ratios = np.linspace(0., 2, 21)
    ratios = np.linspace(0., 0.1, 6)
    ratios = np.append(ratios, np.linspace(0.3, 1.9, 9))
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
        cfg.model['dfn_weight'] = n

        cfg.cfg_name = 'different_fixed_dfn_weight'
        cfg.uid = n
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name,
            'different_fixed_dfn_weight,weight={:.2f},loss={}'.format(
                n, cfg.model['backbone']['loss_cls']['type']))

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None
        cfgs.append(cfg)
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(
        cfg_dir,
        'different_dfn_weight_test,loss={},weight=0.00-2.00,.txt'.format(cfgs[0].model['backbone']['loss_cls']['type']))
    # batch_test(cfgs, save_path, 60 * 2, mode='val')


def baseline_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['garbage.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = 'garbage_baseline'
    cfg.uid = 'mode=baseline'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'garbage_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def anchor_cluster_train():
    from tricks.data_cluster import anchor_cluster
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['garbage.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    # new added
    cfg.model['rpn_head']['anchor_ratios'] = list(anchor_cluster(cfg.data['train']['ann_file'], n=6))

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = 'garbage_baseline'
    cfg.uid = 'anchor_cluster=6'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'garbage_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def larger_lr_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['garbage.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] *= 1.5

    cfg.cfg_name = 'garbage_baseline'
    cfg.uid = 'lr={:.2f}'.format(cfg.optimizer['lr'])
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'garbage_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def twice_epochs_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['garbage.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.total_epochs *= 2

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = 'garbage_baseline'
    cfg.uid = 'epoch=2x'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'garbage_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def the_same_ratio_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['garbage.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.train_pipline[2]['img_scale'] = (1333, 750)
    cfg.test_pipeline[1]['img_scale'] = (1333, 750)

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = 'garbage_baseline'
    cfg.uid = 'img_scale=(1333,750)'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'garbage_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def main():
    # trick 0: baseline
    baseline_train()

    # trick 1: anchor cluster
    anchor_cluster_train()

    # trick 2: larger lr
    larger_lr_train()

    # trick 3: 2x epochs
    twice_epochs_train()

    # trick 4:
    the_same_ratio_train()
    pass


if __name__ == '__main__':
    main()
