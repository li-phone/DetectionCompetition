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
    img_dir = "/home/lifeng/undone-work/DefectNet/tools/data/tile/raw/tile_round1_testA_20201231/testA_imgs"
    post_process_file = "/data/liphone/detcomp/mmdet-v2/tile/baseline_cut_1000x1000_2/baseline_cut_1000x1000_2_do_submit_testA.json"
    save_file = "/home/lifeng/undone-work/DetCompetition/mmdet-v2/work_dirs/tile/baseline_cut_1000x1000_2/post_process_submit_testA_IoU_0_2.json"
    nms = dict(type='nms', iou_threshold=0.2)
    score_thr = 0.1


def post_process():
    config = Config()
    with open(config.post_process_file, "r") as fp:
        results = json.load(fp)
    results = json_normalize(results)
    results = results[results['score'] > config.score_thr]
    save_results = []
    for filename in tqdm(np.unique(results['name'])):
        result = results[results['name'] == filename].sort_values(by='category')
        bboxes = torch.from_numpy(np.array(list(result['bbox']))).float()
        scores = torch.from_numpy(np.array(list(result['score']))).float()
        labels = torch.from_numpy(np.array(list(result['category']))).long()
        bboxes, keep = batched_nms(bboxes, scores, labels, nms_cfg=config.nms)
        labels = labels[keep]
        for r, label in zip(bboxes, labels):
            bbox = list(map(float, r[:4]))
            category_id, score = int(label), r[4]
            save_results.append({'name': str(filename), 'category': int(category_id),
                                 'bbox': bbox, 'score': float(score)})
        # from mmcv.visualization.image import imshow_det_bboxes
        # anns = np.array(result['bbox'])[list(keep)]
        # score = np.array(result['score'])[list(keep)]
        # bbox = np.array([[b[0], b[1], b[2], b[3], score[i]] for i, b in enumerate(anns)])
        # img = imshow_det_bboxes(os.path.join(config.img_dir, filename), bbox, labels, show=False)
        # cv2.imwrite(filename, img)
    with open(config.save_file, "w") as fp:
        json.dump(save_results, fp, indent=4, ensure_ascii=False)
    print("process ok!")


if __name__ == '__main__':
    post_process()
