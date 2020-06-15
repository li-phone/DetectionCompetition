# batch train
import os
import time
import mmcv
import copy
import json
import numpy as np
import os.path as osp
from pycocotools.coco import COCO
from batch_process import batch_train, batch_test

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


class BatchTrain(object):
    def __init__(self, cfg_path, data_mode='val', train_sleep_time=60, test_sleep_time=60 * 2):
        self.cfg_path = cfg_path
        self.cfg_dir = cfg_path[:cfg_path.rfind('/')]
        self.cfg_name = os.path.basename(cfg_path).split('.py')[0]
        self.data_mode = data_mode
        self.train_sleep_time = train_sleep_time
        self.test_sleep_time = test_sleep_time

    def common_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)
        cfg.first_model_cfg = None
        cfg.cfg_name = str(self.cfg_name)
        cfg.uid = str(self.cfg_name)
        if cfg.resume_from is None:
            cfg.resume_from = os.path.join(cfg.work_dir, 'epoch_12.pth')
            if not os.path.exists(cfg.resume_from):
                cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(cfg.work_dir, str(self.cfg_name) + '_{}.txt'.format(self.data_mode))
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)


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
