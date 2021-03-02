import os
import time
import json
import cv2
import torch
from tqdm import tqdm
import numpy as np
from mmcv.ops.nms import batched_nms
from pandas import json_normalize


class Config(object):
    img_dir = "/home/lifeng/undone-work/DefectNet/tools/data/tile/raw/tile_round1_testB_20210128/testB_imgs"
    post_process_file = "work_dirs/track/submit_testA_1000x1000_bs_r50_20e_track__thr_0.001.json"
    save_file = "work_dirs/track/submit_testA_1000x1000_bs_r50_20e_track__thr_0.15__softnms.json"
    # nms = dict(type='nms', iou_threshold=0.5)
    nms = dict(type='soft_nms', iou_threshold=0.5)
    # nms = dict(score_thr=0.15, nms=dict(type='soft_nms', iou_thr=0.5), max_per_img=200)
    score_thr = 0.15


def post_process():
    config = Config()
    with open(config.post_process_file, "r") as fp:
        results = json.load(fp)
    results = json_normalize(results)
    results = results[results['score'] > config.score_thr]
    save_results = []
    for filename in tqdm(np.unique(results['image_id'])):
        result = results[results['image_id'] == filename].sort_values(by='category_id')
        bboxes = []
        for i in range(len(result)):
            x = result.iloc[i]
            bboxes.append(
                [x['bbox_left'], x['bbox_top'], x['bbox_left'] + x['bbox_width'], x['bbox_top'] + x['bbox_height']])
        bboxes = torch.from_numpy(np.array(bboxes)).float()
        scores = torch.from_numpy(np.array(list(result['score']))).float()
        labels = torch.from_numpy(np.array(list(result['category_id']))).long()
        bboxes, keep = batched_nms(bboxes, scores, labels, nms_cfg=config.nms)
        labels = labels[keep]
        for r, label in zip(bboxes, labels):
            bbox = list(map(float, r[:4]))
            category_id, score = int(label), r[4]
            save_results.append({
                'image_id': int(filename),
                'category_id': int(category_id),
                'bbox_left': round(bbox[0], 2),
                'bbox_top': round(bbox[1], 2),
                'bbox_width': round(bbox[2] - bbox[0], 2),
                'bbox_height': round(bbox[3] - bbox[1], 2),
                'score': round(float(score), 6),
            })
        # from mmcv.visualization.image import imshow_det_bboxes
        # anns = np.array(result['bbox'])[list(keep)]
        # score = np.array(result['score'])[list(keep)]
        # bbox = np.array([[b[0], b[1], b[2], b[3], score[i]] for i, b in enumerate(anns)])
        # img = imshow_det_bboxes(os.path.join(config.img_dir, filename), bbox, labels, show=False)
        # cv2.imwrite(filename, img)
    with open(config.save_file, "w") as fp:
        # json.dump(save_results, fp, indent=4, ensure_ascii=False)
        json.dump(save_results, fp)
    print("process ok!")


if __name__ == '__main__':
    post_process()
