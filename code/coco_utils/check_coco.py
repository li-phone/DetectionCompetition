import os
import json
import numpy as np
import shutil
import pandas as pd
import os.path as osp
from tqdm import tqdm
import glob
import cv2 as cv
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from draw_util import *


def draw_coco(ann_path, img_dir, save_dir=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    coco_api = COCO(ann_path)
    image_ids = coco_api.getImgIds()
    label_list = [0] * len(coco_api.cats)
    for k, v in coco_api.cats.items():
        label_list[int(k)] = v['name']
    colors = get_colors(len(label_list))

    for img_id in tqdm(image_ids):
        image_info = coco_api.loadImgs(img_id)[0]
        ann_ids = coco_api.getAnnIds(imgIds=img_id)
        annotations = coco_api.loadAnns(ann_ids)

        save_name = os.path.join(save_dir, image_info['file_name'])
        if os.path.exists(save_name):
            continue
        img_path = os.path.join(img_dir, image_info['file_name'])
        if not os.path.exists(img_path):
            print(img_path, "not exists")
            continue
        bboxes, labels = [], []
        for ann in annotations:
            bb = ann['bbox']
            bb[2] += bb[0]
            bb[3] += bb[1]
            bboxes.append(bb)
            labels.append(ann['category_id'])
        image = read_img(img_path)
        image = draw_bbox(image, bboxes, labels, label_list, colors)
        save_img(image, save_name)


import json
import os
import cv2
import matplotlib.pyplot as plt


def visualize(image_dir, annotation_file, file_name):
    '''
    Args:
        image_dir (str): image directory
        annotation_file (str): annotation (.json) file path
        file_name (str): target file name (.jpg)
    Returns:
        None
    Example:
        image_dir = "./images"
        annotation_file = "./annotations.json"
        file_name = 'img_0028580.jpg'
        visualize(image_dir, annotation_file, file_name)
    '''
    image_path = os.path.join(image_dir, file_name)
    assert os.path.exists(image_path), "image path not exist."
    assert os.path.exists(annotation_file), "annotation file path not exist"
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    with open(annotation_file) as f:
        data = json.load(f)
    image_id = None
    for i in data['images']:
        if i['file_name'] == file_name:
            image_id = i['id']
            break
    if not image_id:
        print("file name {} not found.".format(file_name))
    for a in data['annotations']:
        if a['image_id'] == image_id:
            bbox = [int(b) for b in a['bbox']]
            bbox[2] = bbox[2] + bbox[0] - 1
            bbox[3] = bbox[3] + bbox[1] - 1
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    plt.imshow(image)
    plt.show()
    return


def check_coco(ann_path, save_path, clear_path=None):
    # if os.path.exists(save_path):
    #     return
    with open(ann_path) as fp:
        coco_api = json.load(fp)

    data_ids = [r['id'] for r in coco_api['images']]
    data_uids = np.unique(data_ids)

    data_ids = [r['id'] for r in coco_api['categories']]
    data_uids = np.unique(data_ids)

    data_ids = [r['id'] for r in coco_api['annotations']]
    data_uids = np.unique(data_ids)

    for i, r in enumerate(coco_api['annotations']):
        r['id'] = i
    with open(save_path, 'w') as fp:
        json.dump(coco_api, fp)
    if clear_path is not None:
        images = {r['id']: r for r in coco_api['images']}
        annotations = [r for r in coco_api['annotations'] if r['category_id'] != 0]
        for r in annotations:
            r['image_id'] = images[r['image_id']]['file_name']
        file_names = {r['image_id']: 1 for r in annotations}
        images = {k: v for k, v in images.items() if v['file_name'] in file_names}
        images = [v for k, v in images.items()]
        name_map = {}
        for i, x in enumerate(images):
            x['id'] = i
            name_map[x['file_name']] = i

        for i, x in enumerate(annotations):
            x['image_id'] = name_map[x['image_id']]

        coco_api['images'] = images
        coco_api['annotations'] = annotations
        with open(clear_path, 'w') as fp:
            json.dump(coco_api, fp)


if __name__ == "__main__":
    ann_path = 'E:/data/images/detections/alcohol/annotations/chongqing1_round1_train1_20191223_annotations.json'
    new_ann_path = 'E:/data/images/detections/alcohol/annotations/instances_train_20191223_annotations.json'
    clear_ann_path = 'E:/data/images/detections/alcohol/annotations/instances_train_20191223_nobg.json'
    img_dir = r'C:\Users\zl\liphone\home\data\detection\alcohol\coco_alcohol\train'
    save_dir = r'C:\Users\zl\liphone\home\data\detection\alcohol\coco_alcohol\draw_gt_results'
    check_coco(ann_path, new_ann_path, clear_ann_path)
    # draw_coco(new_ann_path, img_dir, save_dir)
