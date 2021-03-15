from pandas.io.json import json_normalize
from pandas import json_normalize
import json
import os.path as osp
import numpy as np
from pandas import json_normalize
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from mmcv.visualization.image import imshow_det_bboxes


def plt_img(filename, img_dir="data/track/"):
    coco = COCO(filename)
    imgIds = ["03_Train_Station Square/IMG_03_11__010500_007000_014500_011000.jpg"]
    image = coco.loadImgs(ids=imgIds)[0]
    ann_ids = coco.getAnnIds(imgIds=imgIds)
    anns = coco.loadAnns(ann_ids)
    anns = json_normalize(anns)
    img = osp.join(img_dir, image['file_name'])
    bboxes = np.array(list(anns['bbox']))
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    labels = np.array(list(anns['category_id']))
    img = imshow_det_bboxes(img, bboxes, labels, show=False, thickness=5, bbox_color='blue', )
    plt.imshow(img)
    plt.show()


img_dir = "data/track/trainval/overlap_70_all_category/"
ann_file = "data/track/annotations/overlap_70_all_category/overlap_70_all_category-check.json"
plt_img(ann_file, img_dir)
