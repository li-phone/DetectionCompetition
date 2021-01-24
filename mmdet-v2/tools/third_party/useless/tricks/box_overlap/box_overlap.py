import torch
import numpy as np
from tqdm import tqdm
from pandas import json_normalize
from mmdet.core.bbox.geometry import bbox_overlaps


def C_n_2(s, ind, result):
    if ind == len(s) - 1:
        return result
    for i in range(0, len(s)):
        result.append([s[ind], s[i]])
    return C_n_2(s, ind + 1, result)


def overlap_iou(coco, min_iou=0.2, min_num=10):
    anns = json_normalize(coco['annotations'])
    img_ids = anns['image_id'].unique()
    results = {}
    for img_id in tqdm(img_ids):
        keep_ann = anns[anns['image_id'] == img_id]
        box = [b for b in keep_ann['bbox']]
        box = np.asarray(box)
        box = torch.from_numpy(box)
        box[:, 2] += box[:, 0]
        box[:, 3] += box[:, 1]
        ious = bbox_overlaps(box, box)
        ious = ious.numpy()

        assert ious.shape[0] == keep_ann.shape[0] and ious.shape[0] == ious.shape[1]
        for i in range(ious.shape[0]):
            row_i = keep_ann.iloc[i]
            for j in range(i + 1, ious.shape[1]):
                _iou = ious[i][j]
                if _iou <= min_iou:
                    continue
                row_j = keep_ann.iloc[j]
                if row_i['category_id'] <= row_j['category_id']:
                    key = (row_i['category_id'], row_j['category_id'])
                else:
                    key = (row_j['category_id'], row_i['category_id'])
                if key not in results:
                    results[key] = []
                results[key].append(_iou)
    table = []
    total_iou = []
    cat2label = {v['id']: v['name'] for v in coco['categories']}
    for k, v in results.items():
        table.append(dict(category=(cat2label[k[0]], cat2label[k[1]]), iou=np.mean(v), num=len(v)))
        total_iou.extend(v)
    table = json_normalize(table)
    table = table[table['num'] > min_num]
    table = table.sort_values(by='num', ascending=False)
    print(table)
    print('total overlap iou@{}: {:.3f}'.format(min_iou, np.mean(total_iou)))


def main(anns, min_iou=0.2):
    from pycocotools.coco import COCO
    coco = COCO(anns)
    overlap_iou(coco.dataset, min_iou=min_iou)


if __name__ == '__main__':
    main('/home/liphone/undone-work/data/detection/breast/annotations/instance_train.json', min_iou=0.05)
    # main('E:/liphone/data/images/detections/garbage/annotations/instances_train2017.json', min_iou=0.2)
