# coding=utf-8
from scipy.optimize import minimize
import numpy as np
from mmdet.models.anchor_heads import AnchorHead
from mmdet.core.anchor.anchor_target import anchor_inside_flags
from mmdet.core.bbox import bbox_overlaps
import torch
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def fun(kwargs):
    anchor_len = kwargs['anchor_len']
    featmap_sizes = kwargs['featmap_sizes']
    if 'pos_iou_thr' not in kwargs:
        pos_iou_thr = 0.
    else:
        pos_iou_thr = kwargs['pos_iou_thr']

    cfg = kwargs['cfg']
    if isinstance(cfg, str):
        cfg = mmcv.Config.fromfile(cfg)

    cfg.data.train.test_mode = False
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)

    def f(x):
        anchor_scales, anchor_ratios = list(x[:anchor_len[0]]), list(x[anchor_len[0]:])
        anchor_head = AnchorHead(
            2, 256,
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios)

        ious = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            # if i > 10:
            #     break

            img_metas = data['img_meta'].data[0]
            anchor_list, valid_flag_list, = anchor_head.get_anchors(featmap_sizes, img_metas, 'cpu')

            # concat all level anchors and flags to a single tensor
            num_imgs = len(img_metas)
            for i in range(num_imgs):
                assert len(anchor_list[i]) == len(valid_flag_list[i])
                anchor_list[i] = torch.cat(anchor_list[i])
                valid_flag_list[i] = torch.cat(valid_flag_list[i])
            flat_anchors = anchor_list[0]
            valid_flags = valid_flag_list[0]
            img_meta = img_metas[0]
            inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                               img_meta['img_shape'][:2])
            if not inside_flags.any():
                return 1.0
            # assign gt and sample anchors
            anchors = flat_anchors[inside_flags, :]

            gt_boxes = data['gt_bboxes'].data[0]
            gt_boxes = gt_boxes[0]
            ious_ = bbox_overlaps(gt_boxes, anchors).numpy()
            ious_ = ious_[ious_ >= pos_iou_thr]
            ious.append(np.mean(ious_))

            batch_size = data['img'].data[0].size(0)
            for _ in range(batch_size):
                prog_bar.update()
        iou = np.mean(ious)
        return 1 - iou

    return f


def con(args):
    # 约束条件 分为eq 和ineq
    # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
    x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min}, \
            {'type': 'ineq', 'fun': lambda x: -x[0] + x1max}, \
            {'type': 'ineq', 'fun': lambda x: x[1] - x2min}, \
            {'type': 'ineq', 'fun': lambda x: -x[1] + x2max}, \
            {'type': 'ineq', 'fun': lambda x: x[2] - x3min}, \
            {'type': 'ineq', 'fun': lambda x: -x[2] + x3max}, \
            {'type': 'ineq', 'fun': lambda x: x[3] - x4min}, \
            {'type': 'ineq', 'fun': lambda x: -x[3] + x4max}, \
            )
    return cons


def anchor_optimize(coco, anchor_scales, anchor_ratios, img_w=2446, img_h=1000):
    if isinstance(coco, str):
        from pycocotools.coco import COCO
        coco = COCO(coco)
        coco = coco.dataset
    anchor_len = [len(anchor_scales), len(anchor_ratios)]
    x0 = [s for s in anchor_scales]
    for s in anchor_ratios:
        x0.append(s)
    x0 = np.asarray(x0)

    featmap_sizes = [np.asarray([img_h, img_w]) / 4 / (2 ** i) for i in range(5)]
    featmap_sizes = [torch.Size((int(s[0] + 0.5), int(s[1] + 0.5))) for s in featmap_sizes]
    args = {'featmap_sizes': featmap_sizes, 'anchor_len': anchor_len,
            'cfg': '../../config_alcohol/cascade_rcnn_r50_fpn_1x/fabric.py'}
    # 设置参数范围/约束条件
    args1 = (1., 8. * 2, 0.1, 10., 0.1, 10., 0.1, 10.)  # x1min, x1max, x2min, x2max
    cons = con(args1)
    # 设置初始猜测值

    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)
    print('\n\n')
    print(res.fun)
    print(res.success)
    print(res.x)
    pass


def main():
    scales = [8]
    ratios = [0.5, 1.0, 2.0]
    ann_file = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_train_rate=0.80.json'
    anchor_optimize(ann_file, scales, ratios)
    pass


if __name__ == "__main__":
    main()
    pass
