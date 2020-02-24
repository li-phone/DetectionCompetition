import seaborn as sns
import matplotlib.pyplot as plt
import json
from pandas.io.json import json_normalize
import os
import pandas as pd
import numpy as np
from pycocotools.coco import COCO

sns.set(style="darkgrid")


def count_image(coco, ignore_id=0):
    defect_nums = np.empty(0, dtype=int)
    for image in coco.dataset['images']:
        cnt = 0
        annIds = coco.getAnnIds(imgIds=image['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if ann['category_id'] != ignore_id:
                cnt += 1
        defect_nums = np.append(defect_nums, cnt)
    normal_shape = np.where(defect_nums == 0)[0]
    all_cnt, normal_cnt = len(coco.dataset['images']), normal_shape.shape[0]
    defect_cnt = defect_nums.shape[0] - normal_shape.shape[0]
    assert normal_cnt + defect_cnt == all_cnt
    return all_cnt, normal_cnt, defect_cnt


def coco_summary(coco):
    if isinstance(coco, str):
        coco = COCO(coco)
    img_cnts = count_image(coco)
    print('total images: {}, normal images: {}, defect images: {}, normal : defective: {}\n'
          .format(img_cnts[0], img_cnts[1], img_cnts[2], img_cnts[1] / img_cnts[2]))


def main():
    coco_file = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_train,type=34,.json'
    coco_summary(coco_file)


if __name__ == '__main__':
    main()
