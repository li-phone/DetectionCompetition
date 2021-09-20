import os
import time
import cv2
import torch
import glob
import json
import torch
import numpy as np
from PIL import Image
from pandas import json_normalize
from pandas.io.json import json_normalize
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmcv.ops.nms import batched_nms
from mmdet.third_party.parallel import Parallel
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


class Config(object):
    # data module
    img_dir = "data/orange2/test_A-slice_1000x1000_overlap_0.5"
    slice_num = 10000
    nms_whole = False
    label2name = {0: 0, 1: 1}
    # inference module
    device = 'cuda:0'

    iou_thr = 0.5
    score_thr = 0.8

    def __init__(self, cfg_file=None, ckpt_file=None, ann_file=None):
        self.tasks = glob.glob(self.img_dir + "/*")
        self.config_file = cfg_file
        self.ckpt_file = ckpt_file
        self.ann_file = ann_file
        self.save_file = "__work_dirs__/orange2/detection-results"
        self.model = init_detector(cfg_file, self.ckpt_file, device=self.device)


def mkdirs(path, is_file=True):
    if is_file:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def plt_bbox(image, boxes, labels, idx=None, threshold=0.6):
    # show test image
    import matplotlib.pyplot as plt
    from mmcv.visualization.image import imshow_det_bboxes
    if isinstance(image, str):
        img = cv2.imread(image)
    img = np.array(img)
    index = np.where(boxes[:, 4] >= threshold)
    boxes = boxes[index]
    labels = labels[index]
    img = imshow_det_bboxes(img, boxes[:, :4], labels, show=False)
    # plt.imshow(img)
    file_name = "__show_img__/" + '{}'.format(image) + '.jpg'
    mkdirs(file_name)
    img = np.array(img)
    cv2.imwrite(file_name, img)
    pass


def process(image, **kwargs):
    save_results = dict(result=[])
    config = kwargs['config']
    img_file = image
    bbox_result = inference_detector(config.model, img_file)
    win_bboxes = np.empty([0, 6], dtype=np.float32)
    for j in range(len(bbox_result)):
        if len(bbox_result[j]) <= 0:
            continue
        x = np.array([[j] * len(bbox_result[j])])
        bbox_result[j] = np.concatenate([bbox_result[j], x.T], axis=1)
        win_bboxes = np.append(win_bboxes, bbox_result[j], axis=0)
    keep = np.argsort(-win_bboxes[:, 4])[:config.slice_num]
    win_bboxes = win_bboxes[keep]
    plt_bbox(img_file, win_bboxes, win_bboxes[:, 5])
    if len(win_bboxes) < 1:
        return save_results
    if config.nms_whole:
        mst_bboxes = torch.from_numpy(win_bboxes).float().cuda()
        bboxes = mst_bboxes[:, :4].contiguous()
        scores = mst_bboxes[:, 4].contiguous()
        labels = (mst_bboxes[:, 5].long()).contiguous()
        bboxes, keep = batched_nms(
            bboxes, scores, labels, nms_cfg=config.model.cfg.test_cfg.rcnn.nms)
        labels = labels[keep]
        bboxes = bboxes.cpu().numpy()
    else:
        bboxes = win_bboxes[:, :5]
        labels = win_bboxes[:, 5]
    img_name = os.path.basename(image)
    save_results['result'].append({img_name: dict(bbox=bboxes, label=labels)})
    end = time.time()
    print('second/img: {:.2f}'.format(end - kwargs['time']['start']))
    kwargs['time']['start'] = end
    return save_results


def parallel_infer(cfg_file=None, ckpt_file=None, ann_file=None, save_name=None):
    from x2coco import _get_box
    config = Config(cfg_file, ckpt_file, ann_file)
    mkdirs(config.save_file, is_file=False)
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(
        tasks=config.tasks,
        process=process,
        process_params=process_params,
        collect=['result'],
        workers_num=2,
        print_process=100)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {} s'.format(end - start))
    with open(ann_file) as fp:
        coco = json.load(fp)
    for r1 in results['result']:
        for img_id, pred_rst in r1.items():
            if os.path.exists(os.path.join(config.img_dir, img_id)):
                img_ = Image.open(os.path.join(config.img_dir, img_id))
                height_, width_, _ = img_.height, img_.width, 3
            assert height_ is not None and width_ is not None
            has_object = 0
            coco['images'].append(
                dict(file_name=img_id, id=img_id, width=width_, height=height_, has_object=has_object))
            index = np.where(pred_rst['bbox'][:, 4] >= config.score_thr)
            if len(index) == 0:
                print(f'{img_id} has no predict!')
            bboxes = pred_rst['bbox'][index]
            labels = pred_rst['label'][index]
            for _bbox, label in zip(bboxes, labels):
                bbox = _bbox[:4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                ann_id = len(coco['annotations']) + 1
                ann = dict(
                    id=ann_id,
                    image_id=img_id,
                    category_id=int(label + 1),
                    bbox=_get_box(points),
                    iscrowd=0,
                    ignore=0,
                    area=area
                )
                coco['annotations'].append(ann)
    with open(save_name, 'w') as fp:
        json.dump(coco, fp)
    print("process ok!")
    # 更新数据集


def main(cfg_file=None, ckpt_file=None, ann_file=None, save_name=None):
    parallel_infer(cfg_file, ckpt_file, ann_file, save_name)


def main2():
    cfg_path = "../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0.py"
    load_from = 'work_dirs/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0/epoch_12.pth'
    ann_file = 'data/orange2/annotations/slice_resize_0.5_1.0-train.json'
    save_name = f'data/orange2/annotations/pseudo_iou_0.5_score_0.8.json'
    main(cfg_path, load_from, ann_file, save_name)


if __name__ == '__main__':
    pass
