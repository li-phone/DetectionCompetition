from train import main as train_main
from utils import *

if __name__ == '__main__':
    pcfg = import_module("cfg.py")
    cfg = import_module(pcfg.dataset_cfg_path)
    loss_types = ['CrossEntropyLoss',]
    work_dir = cfg.work_dir
    for type in loss_types:
        cfg.gpus = '1'
        cfg.loss['type'] = type
        cfg.work_dir = work_dir + '/{}'.format(type)
        cfg.resume_from = cfg.work_dir + '/latest.pth'
        train_main(cfg)
