from kfbReader import reader
import numpy as np


class kfb_reader(object):

    def __init__(self, kfb_path, scale=20):
        self.read = reader()
        reader.ReadInfo(self.read, kfb_path, scale, True)
        self.width = self.read.getWidth()
        self.height = self.read.getHeight()
        self.scale = scale

        self.row_idx = 0
        self.col_idx = 0

    def read_kfb_img(self, _bbox, scale=None, mode='xyxy'):
        bbox = _bbox.copy()
        if mode == 'xyxy':
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

        if scale is None:
            scale = self.scale

        roi = self.read.ReadRoi(bbox[0], bbox[1], bbox[2], bbox[3], scale)
        # cv.imshow('roi', roi)
        # cv.waitKey(0)
        return roi

    def traverse_image(self, w, h, n, mode='floor'):
        if mode == 'ceil':
            rows = np.ceil(self.height / h)
            cols = np.ceil(self.width / w)
        elif mode == 'floor':
            rows = np.floor(self.height / h)
            cols = np.floor(self.width / w)

        results = []
        img_cnt = 0
        while self.row_idx < rows:
            while self.col_idx < cols:
                if self.row_idx * h + h >= self.height:
                    h = self.height - self.row_idx * h
                if self.col_idx * w + w >= self.width:
                    w = self.width - self.col_idx * w
                roi_bbox = [self.col_idx * w, self.row_idx * h, w, h]
                results.append((self.read_kfb_img(roi_bbox, self.scale, mode='xywh'), roi_bbox))
                self.col_idx += 1
                img_cnt += 1
                if img_cnt >= n:
                    return results
            self.row_idx += 1
            self.col_idx = 0
        self.row_idx = 0
        self.col_idx = 0
        return results


if __name__ == '__main__':

    import cv2 as cv

    kfb = kfb_reader('G:/cervical/neg/T2019_1.kfb')
    idx = 0
    while True:
        rois = kfb.traverse_image(4000, 2000, 10)
        if len(rois) < 10:
            break
        cv.imwrite('{}.jpg'.format(idx), rois[0][0])
        idx += 1
        # cv.imshow('image', roi)
        # cv.waitKey(0)
