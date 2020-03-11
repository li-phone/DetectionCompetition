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
from batch_test import batch_test
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
        if 'ignore_ids' in cfg.data['test']:
            if cfg.data['test']['ignore_ids'] is None:
                have_bg = False
            else:
                have_bg = True
        else:
            have_bg = False
        infer_params = dict(
            config=cfg,
            resume_from=osp.join(cfg.work_dir, 'epoch_12.pth'),
            infer_object=cfg.data['test']['ann_file'],
            img_dir=cfg.data['test']['img_prefix'],
            work_dir=cfg.work_dir,
            submit_out=osp.join(cfg.work_dir, '{}_submit+epoch_{}.json'.format(cfg_name, 12)),
            have_bg=have_bg,
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


class BatchTrain(object):
    def __init__(self, cfg_path, data_mode='val', train_sleep_time=60, test_sleep_time=60 * 2):
        self.cfg_path = cfg_path
        self.cfg_dir = cfg_path[:cfg_path.rfind('/')]
        self.cfg_name = os.path.basename(cfg_path).split('.py')[0]
        self.data_mode = data_mode
        self.train_sleep_time = train_sleep_time
        self.test_sleep_time = test_sleep_time

    def baseline_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)

        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = str(self.cfg_name) + '_baseline'
        cfg.uid = 'mode=baseline'
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

        # batch_infer(cfgs)

    def multi_scale_train(self, img_scale=None, multiscale_mode='range', ratio_range=None, keep_ratio=True):
        cfg = mmcv.Config.fromfile(self.cfg_path)
        cfg.train_pipeline[2] = mmcv.ConfigDict(
            type='Resize', img_scale=img_scale, ratio_range=ratio_range,
            multiscale_mode=multiscale_mode, keep_ratio=keep_ratio)
        sx = int(np.max([v[0] for v in img_scale]))
        sy = int(np.max([v[1] for v in img_scale]))
        cfg.test_pipeline[1]['img_scale'] = [(sx, sy)]

        cfg.data['train']['pipeline'] = cfg.train_pipeline
        cfg.data['val']['pipeline'] = cfg.test_pipeline
        cfg.data['test']['pipeline'] = cfg.test_pipeline

        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = str(self.cfg_name) + '_baseline'
        cfg.uid = 'img_scale={},multiscale_mode={},ratio_range={},keep_ratio={}' \
            .format(str(img_scale), str(multiscale_mode), str(ratio_range), str(keep_ratio))
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

    def backbone_dcn_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)
        cfg.model['backbone']['dcn'] = dict(  # 在最后三个block加入可变形卷积
            modulated=False, deformable_groups=1, fallback_on_stride=False)
        cfg.model['backbone']['stage_with_dcn'] = (False, True, True, True)

        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = cfg.cfg_name + '_baseline'
        cfg.uid = 'backbone_dcn=True'
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

    def global_context_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)
        cfg.model['bbox_roi_extractor']['global_context'] = True

        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = str(self.cfg_name) + '_baseline'
        cfg.uid = 'global_context=True'
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

    def anchor_cluster_train(self, anchor_ratios=[0.5, 1.0, 2.0], anchor_scales=[8]):
        cfg = mmcv.Config.fromfile(self.cfg_path)

        cfg.model['rpn_head']['anchor_ratios'] = list(anchor_ratios)
        cfg.model['rpn_head']['anchor_scales'] = list([anchor_scales])

        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = str(self.cfg_name) + '_baseline'
        cfg.uid = 'anchor_cluster={}+k={}'.format(str(anchor_ratios), len(anchor_ratios))
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + '+' + cfg.uid)

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

    def compete_train(
            self, multiscale=None, dcn=None, global_context=None,
            anchor_cluster=None, data_augment=None, soft_nms=True):
        if multiscale is None:
            multiscale = dict(
                enable=True,
                resize_cfg=dict(
                    img_scale=[(4096, 800), (4096, 1200)],
                    ratio_range=None,
                    multiscale_mode='range',
                    keep_ratio=True,
                )
            )
        if dcn is None:
            dcn = dict(enable=False)
        if global_context is None:
            global_context = dict(enable=False)
        if anchor_cluster is None:
            anchor_cluster = dict(enable=False, k=7)
        if data_augment is None:
            data_augment = dict(
                enable=False,
                cfg_enable=[False, False, False, False, False],
                cfg=[
                    dict(type='Mixup'),
                    dict(type='Corrupt', corruption='contrast'),
                    dict(type='MinIoURandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
                ])
        cfg = mmcv.Config.fromfile(self.cfg_path)
        # # add balanced l1 loss, not effective
        # cfg.model['rpn_head']['loss_bbox'] = dict(type='BalancedL1Loss', loss_weight=1.0)
        # for rcnn in cfg.model['bbox_head']:
        #     rcnn['loss_bbox'] = dict(type='BalancedL1Loss', loss_weight=1.0)

        if multiscale['enable'] and multiscale['resize_cfg'] is not None:
            # 0.745 ==> 0.810(+0.065) | (1333x800) ==> (1920x1080)
            # 0.822 ==> 0.828(+0.006) | (1920x1080) ==> (img_scale=[(1920, 1080), (1333, 800)], multiscale_mode='value')
            resize_cfg = multiscale['resize_cfg']
            cfg.train_pipeline[2] = mmcv.ConfigDict(
                type='Resize', img_scale=resize_cfg['img_scale'], ratio_range=resize_cfg['ratio_range'],
                multiscale_mode=resize_cfg['multiscale_mode'], keep_ratio=resize_cfg['keep_ratio'])
            sx = int(np.mean([v[0] for v in resize_cfg['img_scale']]))
            sy = int(np.mean([v[1] for v in resize_cfg['img_scale']]))
            cfg.test_pipeline[1]['img_scale'] = [(sx, sy)]

            cfg.data['train']['pipeline'] = cfg.train_pipeline
            cfg.data['val']['pipeline'] = cfg.test_pipeline
            cfg.data['test']['pipeline'] = cfg.test_pipeline
        if dcn['enable']:
            # 0.810 ==> 0.819(+0.009)
            cfg.model['backbone']['dcn'] = dict(
                modulated=False, deformable_groups=1, fallback_on_stride=False)
            cfg.model['backbone']['stage_with_dcn'] = (False, True, True, True)
        if global_context['enable']:
            # 0.819 ==> 0.822(+0.003)
            # global context
            cfg.model['bbox_roi_extractor']['global_context'] = True
        if anchor_cluster['enable']:
            # 0.828 ==> 0.830(+0.002)
            from tricks.kmeans_anchor_boxes.yolo_kmeans import coco_kmeans
            anchor_ratios = coco_kmeans(cfg.data['train']['ann_file'], k=anchor_cluster['k'])
            cfg.model['rpn_head']['anchor_ratios'] = list(anchor_ratios)
        if data_augment['enable']:
            for i, v in enumerate(data_augment['cfg']):
                if data_augment['cfg_enable'][i]:
                    aug_ = mmcv.ConfigDict(**v)
                    cfg.train_pipeline.insert(4, aug_)
                    # cfg.test_pipeline[1]['transforms'].insert(2, aug_)
            cfg.data['train']['pipeline'] = cfg.train_pipeline
            cfg.data['val']['pipeline'] = cfg.test_pipeline
            cfg.data['test']['pipeline'] = cfg.test_pipeline
        if soft_nms:
            # 0.830 ==> 0.833(+0.003)
            cfg.test_cfg['rcnn'] = mmcv.ConfigDict(
                score_thr=0.001, nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.001), max_per_img=100)
        # 0.???
        # focal loss for rcnn
        # for head in cfg.model['bbox_head']:
        #     head['loss_cls'] = dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0)

        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = str(self.cfg_name) + '_strong_baseline'
        cfg.uid = 'multiscale+soft-nms'
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + '+' + cfg.uid)

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)
        batch_infer(cfgs)

    def joint_train(
            self, multiscale=None, dcn=None, global_context=None,
            anchor_cluster=None, data_augment=None, soft_nms=True):
        if multiscale is None:
            multiscale = dict(
                enable=True,
                resize_cfg=dict(
                    img_scale=[(1920, 864), (1920, 1296)],
                    ratio_range=None,
                    multiscale_mode='range',
                    keep_ratio=True,
                )
            )
        if dcn is None:
            dcn = dict(enable=True)
        if global_context is None:
            global_context = dict(enable=True)
        if anchor_cluster is None:
            anchor_cluster = dict(enable=False, k=7)
        if data_augment is None:
            data_augment = dict(
                enable=[False, False, False, False, False],
                cfg=[
                    dict(type='Mixup'),
                    dict(type='Corrupt', corruption='contrast'),
                    dict(type='MinIoURandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
                ])
        cfg = mmcv.Config.fromfile(self.cfg_path)
        if multiscale['enable'] and multiscale['resize_cfg'] is not None:
            # 0.745 ==> 0.810(+0.065) | (1333x800) ==> (1920x1080)
            # 0.822 ==> 0.828(+0.006) | (1920x1080) ==> (img_scale=[(1920, 1080), (1333, 800)], multiscale_mode='value')
            resize_cfg = multiscale['resize_cfg']
            cfg.train_pipeline[2] = mmcv.ConfigDict(
                type='Resize', img_scale=resize_cfg['img_scale'], ratio_range=resize_cfg['ratio_range'],
                multiscale_mode=resize_cfg['multiscale_mode'], keep_ratio=resize_cfg['keep_ratio'])
            sx = int(np.mean([v[0] for v in resize_cfg['img_scale']]))
            sy = int(np.mean([v[1] for v in resize_cfg['img_scale']]))
            cfg.test_pipeline[1]['img_scale'] = [(sx, sy)]

            cfg.data['train']['pipeline'] = cfg.train_pipeline
            cfg.data['val']['pipeline'] = cfg.test_pipeline
            cfg.data['test']['pipeline'] = cfg.test_pipeline
        if dcn['enable']:
            # 0.810 ==> 0.819(+0.009)
            cfg.model['backbone']['dcn'] = dict(
                modulated=False, deformable_groups=1, fallback_on_stride=False)
            cfg.model['backbone']['stage_with_dcn'] = (False, True, True, True)
        if global_context['enable']:
            # 0.819 ==> 0.822(+0.003)
            # global context
            cfg.model['bbox_roi_extractor']['global_context'] = True
        if anchor_cluster['enable']:
            # 0.828 ==> 0.830(+0.002)
            from tricks.kmeans_anchor_boxes.yolo_kmeans import coco_kmeans
            anchor_ratios = coco_kmeans(cfg.data['train']['ann_file'], k=anchor_cluster['k'])
            cfg.model['rpn_head']['anchor_ratios'] = list(anchor_ratios)
        if data_augment['enable']:
            for i, v in enumerate(data_augment['cfg']):
                if data_augment['enable'][i]:
                    aug_ = mmcv.ConfigDict(**v)
                    cfg.train_pipeline.insert(4, aug_)
                    # cfg.test_pipeline[1]['transforms'].insert(2, aug_)
            cfg.data['train']['pipeline'] = cfg.train_pipeline
            cfg.data['val']['pipeline'] = cfg.test_pipeline
            cfg.data['test']['pipeline'] = cfg.test_pipeline
        if soft_nms:
            # 0.830 ==> 0.833(+0.003)
            cfg.test_cfg['rcnn']['nms'] = dict(type='soft_nms', iou_thr=0.3, min_score=0.05)
        # 0.???
        # focal loss for rcnn
        # for head in cfg.model['bbox_head']:
        #     head['loss_cls'] = dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0)

        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = str(self.cfg_name) + '_baseline'
        cfg.uid = 'mode=joint_train+multiscale={}+dcn={}+global_context={}+anchor_cluster={}+data_augment={}+soft_nms={}'.format(
            multiscale['enable'], dcn['enable'], global_context['enable'], anchor_cluster['enable'],
            data_augment['enable'] != None, soft_nms
        )
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + '+' + cfg.uid)

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)
        batch_infer(cfgs)

    def other_cfg_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)

        cfg.first_model_cfg = None
        cfg.cfg_name = str(self.cfg_name) + '_baseline'
        cfg.uid = str(self.cfg_name)
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, 'garbage' + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)
        batch_infer(cfgs)

    # def iou_thr_train(iou_thrs=[0.5, 0.6, 0.7]):
    #     cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    #     cfg_names = [DATA_NAME + '.py', ]
    #
    #     cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    #     for i, rcnn in enumerate(cfg.train_cfg['rcnn']):
    #         rcnn['assigner']['pos_iou_thr'] = iou_thrs[i]
    #         rcnn['assigner']['neg_iou_thr'] = iou_thrs[i]
    #         rcnn['assigner']['min_pos_iou'] = iou_thrs[i]
    #
    #     cfg.data['imgs_per_gpu'] = 2
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
    #
    #     cfg.cfg_name = DATA_NAME + '_baseline'
    #     cfg.uid = 'rcnn_iou_thrs={}'.format(str(iou_thrs))
    #     cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)
    #
    #     cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    #     if not os.path.exists(cfg.resume_from):
    #         cfg.resume_from = None
    #
    #     cfgs = [cfg]
    #     batch_train(cfgs, sleep_time=self.train_sleep_time)
    #     from batch_test import batch_test
    #     save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    #     batch_test(cfgs, save_path, self.test_sleep_time, mode=DATA_MODE)
    #
    # def larger_lr_train():
    #     cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    #     cfg_names = [DATA_NAME + '.py', ]
    #
    #     cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    #
    #     cfg.data['imgs_per_gpu'] = 2
    #     cfg.optimizer['lr'] *= 1.
    #
    #     cfg.cfg_name = DATA_NAME + '_baseline'
    #     cfg.uid = 'lr={:.2f}'.format(cfg.optimizer['lr'])
    #     cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)
    #
    #     cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    #     if not os.path.exists(cfg.resume_from):
    #         cfg.resume_from = None
    #
    #     cfgs = [cfg]
    #     batch_train(cfgs, sleep_time=self.train_sleep_time)
    #     from batch_test import batch_test
    #     save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    #     batch_test(cfgs, save_path, self.test_sleep_time, mode=DATA_MODE)
    #
    # def twice_epochs_train():
    #     cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    #     cfg_names = [DATA_NAME + '.py', ]
    #
    #     cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    #     cfg.total_epochs *= 2
    #
    #     cfg.data['imgs_per_gpu'] = 2
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
    #
    #     cfg.cfg_name = DATA_NAME + '_baseline'
    #     cfg.uid = 'epoch=2x'
    #     cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)
    #
    #     cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    #     if not os.path.exists(cfg.resume_from):
    #         cfg.resume_from = None
    #
    #     cfgs = [cfg]
    #     batch_train(cfgs, sleep_time=self.train_sleep_time)
    #     from batch_test import batch_test
    #     save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    #     batch_test(cfgs, save_path, self.test_sleep_time, mode=DATA_MODE)
    #
    # def OHEMSampler_train():
    #     cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    #     cfg_names = [DATA_NAME + '.py', ]
    #
    #     cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    #     for rcnn in cfg.train_cfg['rcnn']:
    #         rcnn['sampler'] = dict(
    #             type='OHEMSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True)
    #
    #     cfg.data['imgs_per_gpu'] = 2
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
    #
    #     cfg.cfg_name = DATA_NAME + '_baseline'
    #     cfg.uid = 'sampler=OHEMSampler'
    #     cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)
    #
    #     cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    #     if not os.path.exists(cfg.resume_from):
    #         cfg.resume_from = None
    #
    #     cfgs = [cfg]
    #     batch_train(cfgs, sleep_time=self.train_sleep_time)
    #     from batch_test import batch_test
    #     save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    #     batch_test(cfgs, save_path, self.test_sleep_time, mode=DATA_MODE)
    #
    # def load_pretrain_train():
    #     cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    #     cfg_names = [DATA_NAME + '.py', ]
    #
    #     cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    #     cfg.load_from = ''
    #
    #     cfg.data['imgs_per_gpu'] = 2
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
    #
    #     cfg.cfg_name = DATA_NAME + '_baseline'
    #     cfg.uid = 'load_from={}'.format(os.path.basename(cfg.load_from))
    #     cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)
    #
    #     cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    #     if not os.path.exists(cfg.resume_from):
    #         cfg.resume_from = None
    #
    #     cfgs = [cfg]
    #     batch_train(cfgs, sleep_time=self.train_sleep_time)
    #     from batch_test import batch_test
    #     save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    #     batch_test(cfgs, save_path, self.test_sleep_time, mode=DATA_MODE)
    #
    # def SWA_train():
    #     import torch
    #     import os
    #     import mmcv
    #     from mmdet.models import build_detector
    #
    #     def get_model(config, model_dir):
    #         model = build_detector(config.model, test_cfg=config.test_cfg)
    #         checkpoint = torch.load(model_dir)
    #         state_dict = checkpoint['state_dict']
    #         model.load_state_dict(state_dict, strict=True)
    #         return model
    #
    #     def model_average(modelA, modelB, alpha):
    #         # modelB占比 alpha
    #         for A_param, B_param in zip(modelA.parameters(), modelB.parameters()):
    #             A_param.data = A_param.data * (1 - alpha) + alpha * B_param.data
    #         return modelA
    #
    #     def swa(cfg, epoch_inds, alpha=0.7):
    #         ###########################注意，此py文件没有更新batchnorm层，所以只有在mmdetection默认冻住BN情况下使用，如果训练时BN层被解冻，不应该使用此py　＃＃＃＃＃
    #         #########逻辑上会　score　会高一点不会太多，需要指定的参数是　[config_dir , epoch_indices ,  alpha]　　######################
    #         if isinstance(cfg, str):
    #             config = mmcv.Config.fromfile(cfg)
    #         else:
    #             config = cfg
    #         work_dir = config.work_dir
    #         model_dir_list = [os.path.join(work_dir, 'epoch_{}.pth'.format(epoch)) for epoch in epoch_inds]
    #
    #         model_ensemble = None
    #         for model_dir in model_dir_list:
    #             if model_ensemble is None:
    #                 model_ensemble = get_model(config, model_dir)
    #             else:
    #                 model_fusion = get_model(config, model_dir)
    #                 model_ensemble = model_average(model_ensemble, model_fusion, alpha)
    #
    #         checkpoint = torch.load(model_dir_list[-1])
    #         checkpoint['state_dict'] = model_ensemble.state_dict()
    #         ensemble_path = os.path.join(work_dir, 'epoch_ensemble.pth')
    #         torch.save(checkpoint, ensemble_path)
    #         return ensemble_path
    #
    #     cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    #     cfg_names = [DATA_NAME + '.py', ]
    #
    #     cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    #
    #     cfg.data['imgs_per_gpu'] = 2
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
    #
    #     cfg.cfg_name = DATA_NAME + '_baseline'
    #     cfg.uid = 'SWA=[10,11,12]'
    #
    #     cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + 'mode=baseline')
    #
    #     ensemble_path = swa(cfg, [10, 11, 12])
    #     cfg.resume_from = ensemble_path
    #     if not os.path.exists(cfg.resume_from):
    #         cfg.resume_from = None
    #
    #     cfgs = [cfg]
    #     # batch_train(cfgs, sleep_time=self.train_sleep_time)
    #     from batch_test import batch_test
    #     save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    #     batch_test(cfgs, save_path, self.test_sleep_time, mode=DATA_MODE)
    #
    # def score_thr_train(score_thr=0.02):
    #     cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    #     cfg_names = [DATA_NAME + '.py', ]
    #
    #     cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    #     cfg.test_cfg['rcnn']['score_thr'] = score_thr
    #
    #     cfg.data['imgs_per_gpu'] = 2
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
    #
    #     cfg.cfg_name = DATA_NAME + '_baseline'
    #
    #     cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + 'mode=baseline')
    #     cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    #     if not os.path.exists(cfg.resume_from):
    #         cfg.resume_from = None
    #
    #     cfg.uid = 'score_thr={}'.format(score_thr)
    #     cfgs = [cfg]
    #     # batch_train(cfgs, sleep_time=self.train_sleep_time)
    #     from batch_test import batch_test
    #     save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    #     batch_test(cfgs, save_path, self.test_sleep_time, mode=DATA_MODE)
    #


def main():
    pass


if __name__ == '__main__':
    main()
