from tqdm import tqdm
import glob
import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import random
import pandas as pd

try:
    from pandas import json_normalize
except:
    from pandas.io.json import json_normalize


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


def coco2yolo(ann_file, img_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    targets = []
    for img_id in tqdm(img_ids):
        image_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        img_path = os.path.join(img_dir, image_info['file_name'])
        if not os.path.exists(img_path):
            print(img_path, "not exists")
            continue

        file_name = os.path.join(save_dir, '{}.txt'.format(image_info['file_name'].split('.')[0]))
        with open(file_name, 'w') as fp:
            for ann in annotations:
                bb = ann['bbox']
                bb[2] += bb[0]
                bb[3] += bb[1]
                img_w = image_info['width']
                img_h = image_info['height']
                bb[2] = min(bb[2], img_w)
                bb[3] = min(bb[3], img_h)
                bb = convert((img_w, img_h), bb)
                cls_id = int(ann['category_id']) - 1
                fp.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        targets.append(dict(img=img_path, target=file_name))
    return coco, targets


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='coco2yolo')
    parser.add_argument('--coco',
                        default='/home/liphone/undone-work/data/detection/garbage_huawei/annotations/train.json',
                        help='coco')
    parser.add_argument('--img_dir', default='/home/liphone/undone-work/data/detection/garbage_huawei/images',
                        help='img_dir')
    parser.add_argument('--save_dir', default='/home/liphone/undone-work/data/detection/garbage_huawei/yolo',
                        help='save_dir')
    parser.add_argument('--frac', default=0.8, type=float, help='frac')
    parser.add_argument('--random_state', default=666, help='random_state')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco, targets = coco2yolo(args.coco, args.img_dir, os.path.join(args.save_dir, 'labels'))
    categories = json_normalize(coco.dataset['categories'])
    categories['name'].to_csv(os.path.join(args.save_dir, 'label_list.txt'), index=False, header=False)

    targets = json_normalize(targets)
    targets = targets.sample(frac=1., random_state=args.random_state)
    train_samples = targets.sample(frac=args.frac, random_state=args.random_state)
    val_samples = targets.drop(train_samples.index)
    targets.to_csv(os.path.join(args.save_dir, 'trainval.txt'), index=False, header=False)
    train_samples.to_csv(os.path.join(args.save_dir, 'train.txt'), index=False, header=False)
    val_samples.to_csv(os.path.join(args.save_dir, 'val.txt'), index=False, header=False)


if __name__ == "__main__":
    main()
