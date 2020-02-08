# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval
import mmcv
import copy
import json
import os.path as osp
from defect_test import main as test_main

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


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
    json_txt = rpt_txt[:-4]
    with open(json_txt + '.json', 'a+') as fp:
        jstr = json.dumps(dict(cfg=cfg, mode=mode, data=rpts))
        fp.write(jstr + '\n')


def batch_test(cfgs, save_dir, sleep_time=0):
    save_name = os.path.basename(save_dir)
    save_name = save_name[:save_name.rfind('.')]
    save_dir = save_dir.replace('\\', '/')
    save_dir = save_dir[:save_dir.rfind('/')]
    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))

        eval_test_params = dict(
            config=cfg,
            checkpoint=osp.join(cfg.work_dir, 'latest.pth'),
            json_out=osp.join(cfg.work_dir, save_name + '.json'),
            mode='test',
        )
        report = test_main(**eval_test_params)
        eval_report(osp.join(save_dir, save_name + '.txt'), report, cfg_name, mode='test')
        print('{} eval test successfully!'.format(cfg_name))
        time.sleep(sleep_time)


def different_defect_finding_weight_test():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]

    base_weight = 1.0
    ratios = [0.1 * i for i in range(0, 21, 1)]
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.model['find_weight'] = base_weight * n
        cfg.work_dir += '_{:.1f}x_find_weight'.format(n)
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        cfgs.append(cfg)

    batch_test(cfgs, cfg_dir + '/different_defect_finding_weight_test.txt', 60)


def different_normal_image_ratio_test():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]

    ratios = [0.1 * i for i in range(0, 21, 1)]
    ann_files = ['instance_test_alcohol_normal_ratio_{}.py'.format(i) for i in ratios]
    cfgs = []
    for i, ann_file in enumerate(ann_files):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.model['find_weight'] = 0.1
        cfg.data['test']['ann_file'] = ann_file
        cfg.work_dir += '_{:.1f}x_find_weight'.format(cfg.model['find_weight'])
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        cfgs.append(cfg)

    batch_test(cfgs, cfg_dir + '/different_normal_image_ratio_test.txt', 60)


def main():
    different_defect_finding_weight_test()
    different_normal_image_ratio_test()


if __name__ == '__main__':
    main()
