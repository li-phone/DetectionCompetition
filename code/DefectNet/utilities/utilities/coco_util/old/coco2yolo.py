from tqdm import tqdm
import glob
import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import random


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def coco2yolo(ann_path, IMG_DIR, SAVE_PATH):
    from pycocotools.coco import COCO
    coco_api = COCO(ann_path)
    image_ids = coco_api.getImgIds()

    random.seed(666)
    random.shuffle(image_ids)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for img_id in tqdm(image_ids):
        image_info = coco_api.loadImgs(img_id)[0]
        ann_ids = coco_api.getAnnIds(imgIds=img_id)
        annotations = coco_api.loadAnns(ann_ids)

        img_path = os.path.join(IMG_DIR, image_info['file_name'])
        if not os.path.exists(img_path):
            print(img_path, "not exists")
            continue
        file_name = os.path.join(SAVE_PATH, '{}.txt'.format(image_info['file_name'].split('.')[0]))
        with open(file_name, 'w') as fp:
            for anno in annotations:
                bb = anno['bbox']
                bb[2] += bb[0]
                bb[3] += bb[1]
                img_w = image_info['width']
                img_h = image_info['height']
                bb = convert((img_w, img_h), bb)
                cls_id = int(anno['category_id']) - 1
                fp.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def split_data(save_path, paths, img_dir):
    with open(save_path, 'w') as fp:
        for p in paths:
            img_id = os.path.basename(p).split('.')[0]
            img_p = os.path.join(img_dir, '{}.jpg'.format(img_id))
            if os.path.exists(p) and os.path.exists(img_p):
                fp.write('{}\n'.format(img_p))


if __name__ == "__main__":
    ann_path = r'C:\Users\zl\liphone\home\data\detection\pascalvoc\coco_pascalvoc\annotations\VOCtest_06-Nov-2007_instances_coco_pascalvoc_test.json'
    img_dir = r'C:\Users\zl\liphone\home\data\detection\pascalvoc\coco_pascalvoc\test'
    save_dir = r'C:\Users\zl\liphone\home\data\detection\pascalvoc\coco_pascalvoc\yolo\labels'

    coco2yolo(ann_path, img_dir, save_dir)

    label_paths = glob.glob(save_dir + '\*')
    train_size = int(len(label_paths) * 0.8)
    split_data(
        r'C:\Users\zl\liphone\home\data\detection\pascalvoc\coco_pascalvoc\yolo\trainval.txt',
        label_paths,
        img_dir
    )
    split_data(
        r'C:\Users\zl\liphone\home\data\detection\pascalvoc\coco_pascalvoc\yolo\train.txt',
        label_paths[:train_size],
        img_dir
    )
    split_data(
        r'C:\Users\zl\liphone\home\data\detection\pascalvoc\coco_pascalvoc\yolo\val.txt',
        label_paths[train_size:],
        img_dir
    )
