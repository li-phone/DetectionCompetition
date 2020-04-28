# batch train
from tqdm import tqdm
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
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(cfg.work_dir, str(self.cfg_name) + '_{}.txt'.format(self.data_mode))
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)


def main():
    pass


if __name__ == '__main__':
    main()
