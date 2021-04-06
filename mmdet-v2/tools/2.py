from pandas.io.json import json_normalize
from pandas import json_normalize
import json
import cv2
import os
import os.path as osp
import numpy as np
from pandas import json_normalize
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from mmcv.visualization.image import imshow_det_bboxes


def get_min_max_bbox(df, rate=0.05):
    df = df.sort_values(by='area')
    min_idx = int(len(df) * rate)
    max_idx = int(len(df) * (1. - rate))
    df_min = df.iloc[:min_idx]
    df_max = df.iloc[max_idx:]
    df_min = np.array(list(df_min['bbox']))
    df_max = np.array(list(df_max['bbox']))
    avg_min = np.mean(df_min, axis=0)
    avg_max = np.mean(df_max, axis=0)
    return avg_min, avg_max


def plt_img(filename, img_dir="data/track/"):
    coco = COCO(filename)
    imgIds = ["01_University_Canteen/IMG_01_01__fx_0.250_fy_0.250__005600_000800_006600_001800.jpg"]
    image = coco.loadImgs(ids=imgIds)[0]
    ann_ids = coco.getAnnIds(imgIds=imgIds)
    anns = coco.loadAnns(ann_ids)
    anns = json_normalize(anns)
    # print(get_min_max_bbox(anns[anns['label'] == 'head']))
    # print(get_min_max_bbox(anns[anns['label'] == 'visible body']))
    # print(get_min_max_bbox(anns[anns['label'] == 'full body']))
    # print(get_min_max_bbox(anns[anns['label'] == 'car']))
    img = osp.join(img_dir, image['file_name'])
    bboxes = np.array(list(anns['bbox']))
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    labels = np.array(list(anns['category_id']))
    img = imshow_det_bboxes(img, bboxes, labels, show=False, thickness=5, bbox_color='blue', )
    cv2.imwrite(os.path.basename(image['file_name']), img)
    plt.imshow(img)
    plt.show()


img_dir = "data/track/trainval/mst_slice/"
ann_file = "data/track/annotations/mst_slice/instance_mst_slice-check.json"
plt_img(ann_file, img_dir)
