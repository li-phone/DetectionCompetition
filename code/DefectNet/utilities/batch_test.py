# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval
import mmcv
import copy
import json
import numpy as np
from pycocotools.coco import COCO
import os.path as osp
import pandas as pd
from defect_test import main as test_main

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


def eval_report(rpt_txt, rpts, cfg, uid=None, mode='val'):
    with open(rpt_txt, 'a+') as fp:
        head = '\n\n{} {}, uid={}, mode={} {}\n'.format('=' * 36, cfg, uid, mode, '=' * 36)
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
        if uid is None:
            uid = cfg
        jstr = json.dumps(dict(cfg=cfg, uid=uid, mode=mode, data=rpts))
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
            json_out=osp.join(cfg.work_dir, os.path.basename(cfg.data['test']['ann_file'])),
            mode='test',
        )
        report = test_main(**eval_test_params)
        eval_report(osp.join(save_dir, save_name + '.txt'), report, cfg=cfg_name, uid=cfg.uid, mode='test')
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

    def del_images(dataset, img_ids):
        for img_id in img_ids:
            for i, image in enumerate(dataset['images']):
                if image['id'] == img_id:
                    dataset['images'].pop(i)
            for i, ann in enumerate(dataset['annotations']):
                if ann['image_id'] == img_id:
                    dataset['annotations'].pop(i)
        return dataset

    def cls_dataset(coco):
        if isinstance(coco, str):
            coco = COCO(coco)
        normal_ids, defect_ids = [], []
        for image in coco.dataset['images']:
            img_id = image['id']
            ann_ids = coco.getAnnIds(img_id)
            anns = coco.loadAnns(ann_ids)
            cnt = 0
            for i, ann in enumerate(anns):
                if ann['category_id'] != 0:
                    cnt += 1
            if cnt == 0:
                normal_ids.append(img_id)
            else:
                defect_ids.append(img_id)
        return normal_ids, defect_ids

    coco = COCO('/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.json')
    normal_ids, defect_ids = cls_dataset(coco)
    defect_ids = pd.DataFrame({'id': defect_ids})
    dataset = coco.dataset

    ann_files, uids = [], []
    choose_num = int(defect_ids.shape[0] * 0.05)
    save_json_dir = '/home/liphone/undone-work/data/detection/alcohol/annotations/normal_ratios'
    if not os.path.exists(save_json_dir):
        os.makedirs(save_json_dir)
    while defect_ids.shape[0] > 0:
        ratio = len(normal_ids) / defect_ids.shape[0]
        uids.append(ratio)
        fn = os.path.join(save_json_dir, 'test_set_normal_ratio={}.json'.format(ratio))
        with open(fn, 'w')as fp:
            json.dump(dataset, fp)
        ann_files.append(fn)

        img_ids = defect_ids.sample(n=min(choose_num, defect_ids.shape[0]), random_state=666)
        dataset = del_images(dataset, np.array(img_ids['id']))
        defect_ids = defect_ids.drop(img_ids.index)

    cfgs = []
    for i, ann_file in enumerate(ann_files):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.model['find_weight'] = 0.1
        cfg.data['test']['ann_file'] = ann_file
        cfg.cfg_name = 'fixed_defectnet_finding_weight'
        cfg.uid = uids[i]
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name, 'fixed_defectnet_finding_weight={:.1f}'.format(cfg.model['find_weight']))
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        cfgs.append(cfg)

    batch_test(cfgs, cfg_dir + '/different_normal_image_ratio_test.txt', 60)


def main():
    # different_defect_finding_weight_test()
    different_normal_image_ratio_test()


if __name__ == '__main__':
    main()
