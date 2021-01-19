import cv2
import numpy as np
import copy
import heapq

from ..builder import PIPELINES


@PIPELINES.register_module()
class SliceROI(object):
    """
        使用canny算法从图像中切割出感兴趣的区域，目前仅支持两种方法：
        1. 先自适应阈值，再查找轮廓；
        2. 先用Canny算法，再用霍夫变换求直线，再求感兴趣区域；
        参数：
            training：是否是训练状态
            padding：在ROI区域外围进行扩充
            threshold：手动阈值或者阈值方法
            method：切割ROI方法
        返回：
            返回单个results
    """

    def __init__(self, training=True, padding=50, threshold='ostu', method='findContours'):
        self.training = training
        self.threshold = threshold
        self.method = method
        self.padding = padding

    @staticmethod
    def get_intersection(a, b):
        y = (a[1] * np.cos(b[0]) - b[1] * np.cos(a[0])) / (np.sin(a[0]) * np.cos(b[0]) - np.sin(b[0]) * np.cos(a[0]))
        x = (a[1] * np.sin(b[0]) - b[1] * np.sin(a[0])) / (np.cos(a[0]) * np.sin(b[0]) - np.cos(b[0]) * np.sin(a[0]))
        return (x, y)

    @staticmethod
    def lines2rect(lines, min_angle=2):
        res = {}
        for x1, y1, x2, y2 in lines[:]:
            radian = np.arctan((x1 - x2) / (y2 - y1))
            if np.isnan(radian):
                continue
            dist = x1 * np.cos(radian) + y1 * np.sin(radian)
            th = int((radian * 180 / np.pi) // min_angle)
            if th not in res:
                res[th] = []
            res[th].append([radian, dist])
        res_counter = [[len(v), k] for k, v in res.items()]
        topk = heapq.nlargest(2, res_counter, key=lambda x: x)
        if len(topk) < 2:
            return None, None
        min_k, max_k = topk[0][1], topk[1][1]
        r1, r2 = np.array(res[min_k]), np.array(res[max_k])
        r1_min_idx, r1_max_idx = np.argmin(r1[:, 1]), np.argmax(r1[:, 1])
        r2_min_idx, r2_max_idx = np.argmin(r2[:, 1]), np.argmax(r2[:, 1])
        l, r, t, b = r1[r1_min_idx], r1[r1_max_idx], r2[r2_min_idx], r2[r2_max_idx]
        if l is None or r is None or t is None or b is None:
            return None, None
        p1 = SliceROI.get_intersection(l, t)
        p2 = SliceROI.get_intersection(t, r)
        p3 = SliceROI.get_intersection(r, b)
        p4 = SliceROI.get_intersection(b, l)
        rect = (min(p1[0], p2[0], p3[0], p4[0]), min(p1[1], p2[1], p3[1], p4[1]),
                max(p1[0], p2[0], p3[0], p4[0]), max(p1[1], p2[1], p3[1], p4[1]))
        if sum(np.isnan(rect)) or sum(np.isinf(rect)):
            return None, None
        return rect, (p1, p2, p3, p4)

    @staticmethod
    def cut_max_rect(image, threshold='ostu', method='findContours', **kwargs):
        if isinstance(image, str):
            image = cv2.imread(image)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if isinstance(threshold, str) and threshold == 'ostu':
            thr, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            # from uuid import uuid4
            # cv2.imwrite("tmp/{}_.jpg".format(uuid4()), img)
            # plt.imshow(img)
            # plt.show()
        elif str(threshold).isdigit():
            thr = float(threshold)
        else:
            thr = 50
        if method == 'findContours':
            if threshold != 'ostu':
                print('Warning!!! findContours method must ostu threshold!')
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) < 1:
                return None, None
            max_size, max_area = 0, 0
            size_idx, area_idx = 0, 0
            for i in range(len(contours)):
                if len(contours[size_idx]) < len(contours[i]):
                    size_idx = i
                    max_size = len(contours[i])
                area_ = cv2.contourArea(contours[i])
                if area_ > max_area:
                    area_idx = i
                    max_area = area_
            x, y, w, h = cv2.boundingRect(contours[area_idx])
            rect = [x, y, x + w, y + h]
            # kwargs['area'].append(max_area)
            # cv2.drawContours(image, contours, area_idx, (0, 255, 0), thickness=15)
            # rect = [int(x) for x in rect]
            # cv2.rectangle(image, tuple(rect[:2]), tuple(rect[2:]), (255, 0, 0), thickness=10)
            # plt.imshow(image)
            # plt.show()
            return rect, None
        elif method == 'HoughLinesP':
            canny_img = cv2.Canny(img, thr, 255)
            h, w = canny_img.shape
            minLineLength = int(min(w, h) / 2)
            maxLineGap = int(np.sqrt(w * w + h * h))
            lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, int(minLineLength / 10),
                                    minLineLength=minLineLength, maxLineGap=maxLineGap)
            if lines is None or len(lines) < 1:
                return None, None
            lines = lines[:, 0, :]  # 提取为二维
            rect, pts = SliceROI.lines2rect(lines)
            return rect, pts
        else:
            raise Exception("No such {} implement method!".format(method))

    def __call__(self, results):
        results['slice_roi'] = {'ori_shape': results['img'].shape}
        rect, pts = SliceROI.cut_max_rect(results['img'], self.threshold, self.method)
        if rect is None:
            return None
        if self.padding:
            ori_h, ori_w, ori_c = results['img'].shape
            rect = [max(0, rect[0] - self.padding), max(0, rect[1] - self.padding),
                    min(ori_w - 1, rect[2] + self.padding), min(ori_h - 1, rect[3] + self.padding), ]

        x1, y1, x2, y2 = [int(x) for x in rect]
        img = results['img'][y1:y2, x1:x2, :]
        results['slice_roi']['left_top'] = [rect[0], rect[1]]
        results['slice_roi__left_top'] = [rect[0], rect[1]]
        results['img_info']['height'] = img.shape[0]
        results['img_info']['width'] = img.shape[1]
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        if self.training:
            results['ann_info']['bboxes'][:, 0] -= results['slice_roi__left_top'][0]
            results['ann_info']['bboxes'][:, 2] -= results['slice_roi__left_top'][0]
            results['ann_info']['bboxes'][:, 1] -= results['slice_roi__left_top'][1]
            results['ann_info']['bboxes'][:, 3] -= results['slice_roi__left_top'][1]
            results['gt_bboxes'] = results['ann_info']['bboxes']
        # # test draw cut roi
        # from mmdet.third_party.draw_box import DrawBox
        # import os
        # drawBox = DrawBox(color_num=7)
        # image = drawBox.draw_box(results['img'], results['gt_bboxes'], results['gt_labels'])
        # image = np.array(image)
        # cv2.imwrite("tmp/{}".format(os.path.basename(results['filename'])), image)
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(method={})'.format(
            self.method)


@PIPELINES.register_module()
class SliceImage(object):
    """
        使用滑动窗口方法从图像中切割成若干张小图片：
        参数：
            training：是否是训练状态
            window：滑动窗口的大小
            step：滑动窗口的步长
            order_index：是否按顺序返回切割成若干张小图片的索引，否则返回全部图片
            is_keep_none：是否保留没有bbox的小图片
        返回：
            返回单个results或者若干个results [list]
    """

    def __init__(self, training=True, window=(2666, 1600), step=(1333, 800), order_index=True, is_keep_none=False):
        self.training = training
        self.window = window
        self.step = step
        self.order_index = {} if order_index else None
        self.is_keep_none = is_keep_none

    def __call__(self, results):
        img_h, img_w, _ = results['img'].shape
        cut_results = []
        for i in range(0, img_h, self.step[1]):
            for j in range(0, img_w, self.step[0]):
                h1 = min(self.window[1], img_h - i)
                w1 = min(self.window[0], img_w - j)
                result = copy.deepcopy(results)
                for k, v in result.items():
                    result[k] = copy.deepcopy(results[k])

                result['slice_image'] = {'ori_shape': [img_h, img_w, _]}
                result['slice_image']['left_top'] = [j, i]
                result['slice_image__left_top'] = [j, i]
                is_cut_img = True
                if self.training:
                    result['ann_info']['bboxes'][:, 0] -= result['slice_image__left_top'][0]
                    result['ann_info']['bboxes'][:, 2] -= result['slice_image__left_top'][0]
                    result['ann_info']['bboxes'][:, 1] -= result['slice_image__left_top'][1]
                    result['ann_info']['bboxes'][:, 3] -= result['slice_image__left_top'][1]
                    bboxes = result['ann_info']['bboxes']
                    keep_idx = [i for i in range(len(bboxes)) if bboxes[i][0] >= 0
                                and bboxes[i][1] >= 0 and bboxes[i][2] < w1 and bboxes[i][3] < h1]
                    result['ann_info']['bboxes'] = result['ann_info']['bboxes'][keep_idx]
                    result['ann_info']['labels'] = result['ann_info']['labels'][keep_idx]
                    # result['ann_info']['bboxes_ignore'] = result['ann_info']['bboxes_ignore'][keep_idx]
                    result['gt_bboxes'] = result['ann_info']['bboxes']
                    result['gt_labels'] = result['ann_info']['labels']
                    # result['gt_bboxes_ignore'] = result['gt_bboxes_ignore'][keep_idx]
                    if len(result['gt_bboxes']) <= 0 and len(result['gt_labels']) <= 0:
                        is_cut_img = False
                if self.is_keep_none or is_cut_img:
                    h = min(img_h, i + self.window[1])
                    w = min(img_w, j + self.window[0])
                    img = results['img'][i:h, j:w, :]
                    h2, w2, c2 = img.shape
                    assert h1 == h2 and w1 == w2
                    result['img_info']['height'] = img.shape[0]
                    result['img_info']['width'] = img.shape[1]
                    result['img'] = img
                    result['img_shape'] = img.shape
                    result['ori_shape'] = img.shape
                    cut_results.append(result)
                    # from mmdet.third_party.draw_box import DrawBox
                    # import os
                    # drawBox = DrawBox(color_num=7)
                    # image = drawBox.draw_box(result['img'], result['gt_bboxes'], result['gt_labels'])
                    # image = np.array(image)
                    # cv2.imwrite("tmp/{}_i{}_j{}.jpg".format(os.path.basename(result['filename']), str(i), str(j)),
                    #             image)
                    # import matplotlib.pyplot as plt
                    # plt.imshow(image)
                    # plt.show()
        if len(cut_results) < 1:
            return None
        if self.order_index is not None:
            key = results['filename']
            if key not in self.order_index:
                self.order_index[key] = 0
            else:
                self.order_index[key] = (self.order_index[key] + 1) % len(cut_results)
            return cut_results[self.order_index[key]]
        return cut_results

    def __repr__(self):
        return self.__class__.__name__ + '(window={})'.format(
            self.window)
