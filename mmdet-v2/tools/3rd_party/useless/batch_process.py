# batch train
import os
import time
import mmcv
import copy
import json
from tqdm import tqdm
import numpy as np
import os.path as osp
from pycocotools.coco import COCO
from train import main as train_main
from test import main as test_main

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


def batch_train(cfgs, sleep_time=0, detector=True):
    for cfg in tqdm(cfgs):
        print('\ncfg: {}'.format(cfg.work_dir))
        # train
        if sleep_time >= 0:
            train_params = dict(config=cfg)
            train_main(**train_params)
            time.sleep(sleep_time)
        print('{} train done!'.format(cfg.work_dir))


def batch_test(cfgs, sleep_time=0, mode='test'):
    for i, cfg in tqdm(enumerate(cfgs)):
        print('\ncfg: {}'.format(cfg.work_dir))

        if os.path.exists(osp.join(cfg.work_dir, 'latest.pth')):
            latest_pth = osp.join(cfg.work_dir, 'latest.pth')
        else:
            latest_pth = osp.join(cfg.work_dir, 'epoch_12.pth')

        test_params = dict(
            config=cfg,
            checkpoint=latest_pth,
            format_only=True,
            options=dict(jsonfile_prefix='data_mode={}'.format(mode)),
            mode=mode,
        )
        report = test_main(**test_params)
        print('{} eval test done!'.format(cfg.work_dir))
        time.sleep(sleep_time)


class BatchTrain(object):
    def __init__(self, cfg_path, data_mode='val', train_time=60, test_time=60):
        self.cfg_path = cfg_path
        self.cfg_dir = cfg_path[:cfg_path.rfind('/')]
        self.cfg_name = os.path.basename(cfg_path).split('.py')[0]
        self.data_mode = data_mode
        self.train_time = train_time
        self.test_time = test_time

    def common_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)
        if cfg.resume_from is None:
            cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
            if not os.path.exists(cfg.resume_from):
                cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_time)
        if self.test_time >= 0:
            batch_test(cfgs, self.test_time, mode=self.data_mode)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train model by config')
    parser.add_argument('config', help='directory')
    parser.add_argument('--mode', default='test', help='file prefix')
    parser.add_argument('--train_time', default='0', help='file prefix')
    parser.add_argument('--test_time', default='60', help='file prefix')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    BatchTrain(args.config, args.mode, args.train_time, args.test_time).common_train()


if __name__ == '__main__':
    main()
