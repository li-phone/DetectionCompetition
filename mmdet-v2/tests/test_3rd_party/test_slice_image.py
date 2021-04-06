import numpy as np
import requests
import cv2
import time
import os
import torch
import os.path as osp
from pandas import json_normalize
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D


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


def mkdirs(path, is_file=True):
    if is_file:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def imwrite(fname, img):
    img = cv2.resize(img, (img.shape[1] // 10, img.shape[0] // 10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(fname, img)


class TestSliceImage(object):

    @classmethod
    def setup_class(cls):
        # cls.img_file = osp.join(osp.dirname(__file__), '../data/slice_sample.jpg')
        # cls.img = cv2.imread(cls.img_file)
        # assert cls.img is not None
        # img = cv2.resize(cls.img, (cls.img.shape[1] // 10, cls.img.shape[0] // 10), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(cls.img_file[:-4] + '_XS.jpg', img)
        pass

    def slice(self, img, base_win, img_shape, step=(0., 0.), fx=None, fy=None, center=None):
        img_h, img_w = img_shape[:2]
        center = list(center)
        center[0] = center[0] if center[0] is not None else base_win[0] / 2
        center[1] = center[1] if center[1] is not None else base_win[1] / 2

        def slice_(ctr, window, sx=1., sy=1.):
            cv2.circle(img, (int(ctr[0]), int(ctr[1])), 10, (0, 255, 0), 5)
            cnt = 0
            X, Y = ctr
            window = [window[0], window[1]]
            while 0 <= Y < img_h:
                x = X
                win = [window[0], window[1]]
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                while 0 <= x < img_w:
                    left, top = max(0, x - win[0] / 2), max(0, Y - win[1] / 2)
                    right, bottom = min(img_w, x + win[0] / 2), min(img_h, Y + win[1] / 2)
                    if x - win[0] / 2 < 0:
                        right = left + win[0]
                    if Y - win[1] / 2 < 0:
                        bottom = top + win[1]
                    if x + win[0] / 2 > img_w:
                        left = right - win[0]
                    if Y + win[1] / 2 > img_h:
                        top = bottom - win[1]
                    roi = list(map(int, (left, top, right, bottom)))
                    cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), color=color, thickness=20)
                    x += win[0] * (1. - step[0]) * sx
                    cnt += 1
                    # if fx: win[0] = fx(win[0])
                    # if fy: win[1] = fy(win[1])
                Y += window[1] * (1. - step[1]) * sy
                if fx: window[0] = fx(window[0])
                if fy: window[1] = fy(window[1])
                window = [min(10000, window[0]), min(10000, window[1])]
            return cnt

        _win = [base_win[0], base_win[1]]
        cnt = slice_(center, _win, 1., 1.)
        cnt += slice_(center, _win, 1., -1.)
        print('cnt', cnt)
        return img

    def test_mst_overlap_slice2(self):
        import os
        import glob
        ids = glob.glob("../../tools/data/track/panda_round1_test_202104_A/*/*.jpg")
        have_seen = set()
        for id in ids:
            filename = id
            if os.path.dirname(filename) in have_seen:
                continue
            have_seen.add(os.path.dirname(filename))
            fx = lambda x: x * 1.5
            fy = lambda x: x * 1.5
            img = cv2.imread(filename)
            self.slice(img, (1000, 1000), img.shape[:2], step=(0.25, 0.25), fx=fx, fy=fy,
                       center=(None, img.shape[0] / 3))
            id = 'test_imgs/' + os.path.basename(id)
            mkdirs(id[:-4] + '_mst_overlap_slice.jpg')
            imwrite(id[:-4] + '_mst_overlap_slice.jpg', img)

    def test_mst_overlap_slice(self):
        import os
        import glob
        from pycocotools.coco import COCO
        img_dir = "../../tools/data/track/panda_round1_train_202104_part1/"
        ann_file = "../../tools/data/track/annotations/ori_instance_all_category.json"
        coco = COCO(ann_file)
        ids = set([x['id'] for x in coco.dataset['images']])
        have_seen = set()
        for id in ids:
            filename = os.path.join(img_dir, id)
            if os.path.dirname(filename) in have_seen:
                continue
            have_seen.add(os.path.dirname(filename))
            imgIds = [id]
            ann_ids = coco.getAnnIds(imgIds=imgIds)
            anns = coco.loadAnns(ann_ids)
            anns = json_normalize(anns)
            df_min, df_max = get_min_max_bbox(anns[anns['label'] == 'head'])
            fx = lambda x: x * 1.25
            fy = lambda x: x * 1.25
            img = cv2.imread(filename)
            self.slice(img, (2000, 2000), img.shape[:2], step=(0.25, 0.25), fx=fx, fy=fy, center=(None, df_min[1]))
            id = 'imgs/' + id
            mkdirs(id[:-4] + '_mst_overlap_slice.jpg')
            imwrite(id[:-4] + '_mst_overlap_slice.jpg', img)

    def test_mst_overlap_slice(self):
        import os
        import glob
        from pycocotools.coco import COCO
        img_dir = "../../tools/data/track/panda_round1_train_202104_part1/"
        ann_file = "../../tools/data/track/annotations/ori_instance_all_category.json"
        coco = COCO(ann_file)
        ids = set([x['id'] for x in coco.dataset['images']])
        have_seen = set()
        for id in ids:
            filename = os.path.join(img_dir, id)
            if os.path.dirname(filename) in have_seen:
                continue
            have_seen.add(os.path.dirname(filename))
            imgIds = [id]
            ann_ids = coco.getAnnIds(imgIds=imgIds)
            anns = coco.loadAnns(ann_ids)
            anns = json_normalize(anns)
            df_min, df_max = get_min_max_bbox(anns[anns['label'] == 'head'])
            fx = lambda x: x * 1.25
            fy = lambda x: x * 1.25
            img = cv2.imread(filename)
            self.slice(img, (2000, 2000), img.shape[:2], step=(0.25, 0.25), fx=fx, fy=fy, center=(None, df_min[1]))
            id = 'imgs/' + id
            mkdirs(id[:-4] + '_mst_overlap_slice.jpg')
            imwrite(id[:-4] + '_mst_overlap_slice.jpg', img)

    def test_common_slice(self):
        self.slice((4000, 4000), self.img.shape[:2])
        self.imwrite(self.img_file[:-4] + '_common_slice.jpg')

    def test_common_overlap_slice(self):
        self.slice((4000, 4000), self.img.shape[:2], step=(0.2, 0.2))
        self.imwrite(self.img_file[:-4] + '_overlap_slice.jpg')
