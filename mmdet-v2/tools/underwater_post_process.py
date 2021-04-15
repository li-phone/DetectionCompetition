import os
import time
import json
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from mmcv.ops.nms import batched_nms
from pandas import json_normalize


class Config(object):
    post_process_file = "work_dirs/underwater/best-x101-mst_slice.csv"
    save_file = "work_dirs/underwater/best-x101-mst_slice-soft_nms_2.csv"
    # nms = dict(type='nms', iou_threshold=0.5)
    # nms = dict(score_thr=0.15, nms=dict(type='soft_nms', iou_thr=0.5), max_per_img=200)
    nms = dict(type='soft_nms', iou_threshold=0.5)
    label2name = {0: 'echinus', 1: 'holothurian', 2: 'scallop', 3: 'starfish', 4: 'waterweeds'}
    name2label = {'echinus': 0, 'holothurian': 1, 'scallop': 2, 'starfish': 3, 'waterweeds': 4}
    # nms = None
    score_thr = 0.01
    # score_thr = {
    #     1: 0.01,  # visible body
    #     2: 0.01,  # full body
    #     3: 0.01,  # head
    #     4: 0.01,  # vehicle
    # }


label2name = {0: 'echinus', 1: 'holothurian', 2: 'scallop', 3: 'starfish', 4: 'waterweeds'}


def nms(results, nms_cfg):
    save_results = []
    # results = json_normalize(results)
    for filename in tqdm(np.unique(results['image_id'])):
        result = results[results['image_id'] == filename].sort_values(by='name')
        result = result.sort_values(by='confidence', ascending=False)
        # result = result.iloc[:5000]
        bboxes = []
        for i in range(len(result)):
            x = result.iloc[i]
            bboxes.append([x['xmin'], x['ymin'], x['xmax'], x['ymax']])
        bboxes = torch.from_numpy(np.array(bboxes)).float().cuda()
        scores = torch.from_numpy(np.array(list(result['confidence']))).float().cuda()
        labels = torch.from_numpy(np.array(list(result['name']))).long().cuda()
        bboxes, keep = batched_nms(bboxes, scores, labels, nms_cfg=nms_cfg)
        labels = labels[keep]
        assert len(bboxes) == len(labels)
        for r, label in zip(bboxes, labels):
            bbox = list(map(float, r[:4]))
            category_id, score = int(label), r[4]
            save_results.append({
                'name':
                    label2name[category_id],
                'image_id':
                    str('{:06d}'.format(filename)),
                'xmin':
                    bbox[0],
                'ymin':
                    bbox[1],
                'xmax':
                    bbox[2],
                'ymax':
                    bbox[3],
                'confidence':
                    float(score)
            })
        # from mmcv.visualization.image import imshow_det_bboxes
        # anns = np.array(result['bbox'])[list(keep)]
        # score = np.array(result['score'])[list(keep)]
        # bbox = np.array([[b[0], b[1], b[2], b[3], score[i]] for i, b in enumerate(anns)])
        # img = imshow_det_bboxes(os.path.join(config.img_dir, filename), bbox, labels, show=False)
        # cv2.imwrite(filename, img)
    return save_results


def post_process():
    config = Config()
    save_results = pd.read_csv(config.post_process_file)
    save_results['name'] = save_results['name'].apply(lambda x: config.name2label[x])
    # save_results = save_results[save_results['confidence'] > config.score_thr]
    if config.nms is not None:
        save_results = nms(save_results, config.nms)
    results = json_normalize(save_results)
    results.to_csv(config.save_file, index=False)
    print("process ok!")


if __name__ == '__main__':
    post_process()
