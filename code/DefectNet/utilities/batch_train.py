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


def eval_report(rpt_txt, rpts, cfg, mode='val'):
    with open(rpt_txt, 'a+') as fp:
        head = '\n\n{} {}@{} {}\n'.format('=' * 36, cfg, mode, '=' * 36)
        fp.write(head)
        for k1, v1 in rpts.items():
            fp.write(k1 + ':\n')  # bbox

            fp.write('log' + ':\n')  # log
            for k2, v2 in v1['log'].items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        fp.write('\n' + k3 + ':\n' + str(v3) + '\n')
                else:
                    fp.write('\n' + k2 + ':\n' + str(v2) + '\n')

            fp.write('data' + ':\n')  # log
            for k2, v2 in v1['data'].items():
                fp.write('\n' + k2 + ':\n' + json.dumps(v2) + '\n')
    json_txt = rpt_txt[:-4]
    with open(json_txt + '.json', 'a+') as fp:
        jstr = json.dumps(dict(cfg=cfg, mode=mode, data=rpts))
        fp.write(jstr + '\n')


def main():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]
    # mode_paths = [os.path.join(cfg_dir, m) for m in cfg_names]
    # cfgs = [mmcv.Config.fromfile(p) for p in mode_paths]

    # watch train effects using different base cfg
    base_weight = 1.0
    ns = [0.1 * i for i in range(0, 21, 2)]
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.model['find_weight'] = base_weight * n
        cfg.work_dir += '_{:.1f}x_find_weight'.format(n)
        for v in cfg.model['bbox_head']:
            v['num_classes'] += int(cfg.model['background_train'])
        cfg.data['train']['background_train'] = cfg.model['background_train']
        cfg.data['val']['background_train'] = cfg.model['background_train']
        cfg.data['test']['background_train'] = cfg.model['background_train']
        cfgs.append(cfg)

    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))

        # train
        train_params = dict(config=cfg)
        train_main(**train_params)
        print('{} train successfully!'.format(cfg_name))
        hint()

        # eval for val set
        eval_val_params = dict(
            config=cfg,
            checkpoint=osp.join(cfg.work_dir, 'latest.pth'),
            json_out=osp.join(cfg.work_dir, 'eval_val_set.json'),
            mode='val',
        )
        report = test_main(**eval_val_params)
        eval_report(osp.join(cfg_dir, 'eval_alcohol_dataset_report.txt'), report, cfg_name, mode='val')
        print('{} eval val successfully!'.format(cfg_name))
        hint()

        # eval for test set
        eval_test_params = dict(
            config=cfg,
            checkpoint=osp.join(cfg.work_dir, 'latest.pth'),
            json_out=osp.join(cfg.work_dir, 'eval_test_set.json'),
            mode='test',
        )
        report = test_main(**eval_test_params)
        eval_report(osp.join(cfg_dir, 'eval_alcohol_dataset_report.txt'), report, cfg_name, mode='test')
        print('{} eval test successfully!'.format(cfg_name))
        hint()

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

        # time.sleep(1800)


if __name__ == '__main__':
    main()
