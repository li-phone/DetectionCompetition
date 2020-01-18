import json
from draw_util import *
import pandas as pd
from tqdm import tqdm
import os
from pycocotools.coco import COCO


def coco_res2list(file_path, save_path):
    with open(file_path) as fp:
        results = json.load(fp)
    img_ids = {r['id']: r['file_name'] for r in results['images']}
    for r in results['annotations']:
        r['image_id'] = img_ids[r['image_id']]
    with open(save_path, 'w') as fp:
        json.dump(results['annotations'], fp)


def main():
    img_dir = r'C:\Users\zl\liphone\home\data\detection\alcohol\coco_alcohol\chongqing1_round1_testA_20191223\images'
    save_dir = r'C:\Users\zl\liphone\home\data\detection\alcohol\coco_alcohol\pred'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fname = r'C:\Users\zl\liphone\home\fabric_detection\uselessNet\code\mmdetection\work_dirs\alcohol\cascade_rcnn_r50_fpn_1x\cascade_rcnn_r50_fpn_1x_alcohol_latest_submit.json'
    save_name = 'results.json'
    coco_res2list(fname, save_name)
    results = pd.read_json(save_name)
    results = results[results['score'] >= 0.1]
    defect_name2label = {
        0: '背景',
        1: '瓶盖破损',
        2: '瓶盖变形',
        3: '瓶盖坏边',
        4: '瓶盖打旋',
        5: '瓶盖断点',
        6: '标贴歪斜',
        7: '标贴起皱',
        8: '标贴气泡',
        9: '喷码正常',
        10: '喷码异常'
    }
    label_list = [v for k, v in defect_name2label.items()]
    colors = get_colors(len(label_list))
    img_ids = results['image_id'].unique()
    for img_id in tqdm(img_ids):
        result = results[results['image_id'] == img_id]
        image = read_img(os.path.join(img_dir, img_id))
        img_pred = draw_bbox(image, list(result['bbox']), list(result['category_id']), label_list, colors)
        # save_img(image, os.path.join(save_dir, img_id))
        save_img(img_pred, os.path.join(save_dir, img_id + '_pred.jpg'))


if __name__ == '__main__':
    main()
