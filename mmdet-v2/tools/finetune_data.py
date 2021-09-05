import os
import time
import cv2
import torch
import glob
import json
import torch
import numpy as np
from pandas import json_normalize
from pandas.io.json import json_normalize
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmcv.ops.nms import batched_nms
from mmdet.third_party.parallel import Parallel
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


class Config(object):
    # data module
    img_dir = "data/orange2/slice_1000x1000"
    slice_num = 10000
    nms_whole = False
    label2name = {0: 'bug', 1: 'fruit_bug'}
    # inference module
    device = 'cuda:1'

    iou_thr = 0.9
    score_thr = 0.9

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


def plt_bbox(image, boxes, labels, prefix=None, threshold=0.5):
    # show test image
    import matplotlib.pyplot as plt
    from mmcv.visualization.image import imshow_det_bboxes
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    img = np.array(img)
    index = np.where(boxes[:, 4] >= threshold)
    boxes = boxes[index]
    labels = labels[index]
    img = imshow_det_bboxes(img, boxes[:, :4], labels, show=False)
    # plt.imshow(img)
    if isinstance(image, str):
        file_name = f"__show_img__/finetune-data/{image}.jpg"
    else:
        file_name = f"__show_img__/finetune-data/{prefix}.jpg"
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
    config = Config(cfg_file, ckpt_file, ann_file)
    mkdirs(config.save_file, is_file=False)
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(
        tasks=config.tasks,
        process=process,
        process_params=process_params,
        collect=['result'],
        workers_num=10,
        print_process=100)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {} s'.format(end - start))
    with open(ann_file) as fp:
        coco = json.load(fp)
    anns = json_normalize(coco['annotations'])
    for r1 in results['result']:
        for img_id, pred_rst in r1.items():
            df = anns[anns['image_id'] == img_id]
            bbox1 = [b for b in list(df['bbox'])]
            bbox1 = np.array(bbox1).astype(dtype=np.float)
            bbox1[:, 2] += bbox1[:, 0]
            bbox1[:, 3] += bbox1[:, 1]
            bbox2 = pred_rst['bbox'][:, :4]
            score = pred_rst['bbox'][:, 4]
            bbox1 = torch.from_numpy(bbox1)
            bbox2 = torch.from_numpy(bbox2)
            ious = bbox_overlaps(bbox1, bbox2)
            bbox1 = bbox1.numpy()
            bbox2 = bbox2.numpy()
            ious = ious.numpy()
            for i in range(ious.shape[0]):
                iou = ious[i, :]
                idx = np.argmax(iou)
                if iou[idx] >= config.iou_thr and score[idx] >= config.score_thr:
                    b2 = bbox2[idx]
                    b3 = (bbox1[i] + b2) / 2
                    b3 = [b3[0], b3[1], b3[2] - b3[0], b3[3] - b3[1]]
                    bbox = [float(_) for _ in b3]
                    r1 = df.iloc[i]
                    anns.at[r1['id'], 'bbox'] = bbox
    anns = anns.to_dict(orient='records')
    coco['annotations'] = anns
    with open(save_name, 'w') as fp:
        json.dump(coco, fp)
    print("process ok!")
    # 更新数据集


def main(cfg_file=None, ckpt_file=None, ann_file=None, save_name=None):
    parallel_infer(cfg_file, ckpt_file, ann_file, save_name)


if __name__ == '__main__':
    # cfg_path = "../configs/orange/cas_r50-best-finetune_data.py"
    # load_from = 'work_dirs/cas_r50-best-finetune_data/latest.pth'
    # ann_file = f'data/orange/annotations/instance-train-finetune-{1}.json'
    # save_name = f'data/orange/annotations/instance-train-best-iou_0.8_score_0.8_{1}.json'
    cfg_path = "../configs/orange2/cas_r50-best-slice_1000x1000.py"
    load_from = 'work_dirs/cas_r50-best-slice_1000x1000/latest.pth'
    ann_file = f'data/orange2/annotations/slice_1000x1000_train.json'
    save_name = f'data/orange2/annotations/slice_1000x1000-iou_0.9_score_0.9_iter_{1}.json'
    main(cfg_path, load_from, ann_file, save_name)
