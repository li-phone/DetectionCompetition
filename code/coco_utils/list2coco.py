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

labels2ids = {
    'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
    'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
    'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20,
}


class Convert2COCO:

    def __init__(self, mode="train", info=""):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode = mode
        self.scale = 20
        self.info = info

    def _init_categories(self):
        # for v in range(1, 21):
        #     print(v)
        #     category = {}
        #     category['id'] = v
        #     category['name'] = str(v)
        #     category['supercategory'] = 'defect_name'
        #     self.categories.append(category)
        for k, v in labels2ids.items():
            category = {}
            category['id'] = v
            category['name'] = k
            category['supercategory'] = self.info
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = int(h)
        image['width'] = int(w)
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label, bbox, iscrowd=0):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = int(iscrowd)
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        if not os.path.exists(img_path):
            pass

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))

    def to_coco(self, anno_path, img_dir):
        self._init_categories()
        annos_df = pd.read_json(anno_path)
        annos_df = annos_df.head(50)
        file_names = annos_df['file_name'].unique()
        for file_name in tqdm(file_names):
            img_path = os.path.join(img_dir, file_name)
            if not os.path.exists(img_path):
                print(img_path, "not exists")
                continue
            annos = annos_df[annos_df['file_name'] == file_name]
            fname = annos['file_name'].unique()
            assert fname == file_name
            img_w, img_h = 0, 0
            for idx in range(annos.shape[0]):
                r = annos.iloc[idx]
                img_w, img_h = r['img_w'], r['img_h']
                b = list((int(r['x']), int(r['y']), int(r['w']), int(r['h'])))
                b[2] = b[2] + b[0]
                b[3] = b[3] + b[1]
                assert r['label_name'] in labels2ids
                cls_id = labels2ids[r['label_name']]
                annotation = self._annotation(cls_id, b, r['iscrowd'])
                self.annotations.append(annotation)
                self.ann_id += 1
            self.images.append(self._image(img_path, img_h, img_w))
            # self._cp_img(img_path)
            self.img_id += 1

        instance = {}
        instance['info'] = self.info
        instance['license'] = ['MIT']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance


def _test_convert2coco(anno_path, img_dir, save_path):
    convert2coco = Convert2COCO(info="coco")
    instance = convert2coco.to_coco(anno_path, img_dir)
    convert2coco.save_coco_json(instance, save_path)


if __name__ == "__main__":
    save_dir = '../pascalvoc/coco_pascalvoc/annotations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_dir = "../pascalvoc/coco_pascalvoc/test"
    _test_convert2coco(
        "../pascalvoc/coco_pascalvoc/annotations/VOCtest_06-Nov-2007_pascalvoc_list_train.json",
        img_dir,
        # os.path.join(save_dir, "VOCtest_06-Nov-2007_instances_coco_{}.json".format("pascalvoc_train")),
        os.path.join(save_dir, "small_sample_instances_coco_{}.json".format("train")),
    )

    _test_convert2coco(
        "../pascalvoc/coco_pascalvoc/annotations/VOCtest_06-Nov-2007_pascalvoc_list_val.json",
        img_dir,
        # os.path.join(save_dir, "VOCtest_06-Nov-2007_instances_coco_{}.json".format("pascalvoc_val")),
        os.path.join(save_dir, "small_sample_instances_coco_{}.json".format("val")),
    )

    _test_convert2coco(
        "../pascalvoc/coco_pascalvoc/annotations/VOCtest_06-Nov-2007_pascalvoc_list_test.json",
        img_dir,
        # os.path.join(save_dir, "VOCtest_06-Nov-2007_instances_coco_{}.json".format("pascalvoc_test")),
        os.path.join(save_dir, "small_sample_instances_coco_{}.json".format("test")),
    )
