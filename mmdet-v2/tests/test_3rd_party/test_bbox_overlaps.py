import numpy as np
import requests
import cv2
import time
import torch
import os.path as osp
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D


class TestBBoxOverlaps(object):

    @classmethod
    def setup_class(cls):
        cls.test_size = 20
        cls.test_range = 1
        cls.init_tasks = [cls.test_range for i in range(cls.test_size)]
        cls.img_file = osp.join(osp.dirname(__file__), '../color.jpg')

    def test_simple_roinet(self):
        bboxes1 = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [32, 32, 38, 42],
        ])
        bboxes2 = torch.FloatTensor([
            [0, 0, 10, 20],
            [0, 10, 10, 19],
            [10, 10, 20, 20],
        ])
        bboxes1 = torch.rand((40, 4)) * 1000
        bboxes2 = torch.rand((400000, 4)) * 1000

        bbox_overlaps1 = BboxOverlaps2D()
        bbox_overlaps2 = BboxOverlaps2D(dtype='cpu')
        bbox_overlaps3 = BboxOverlaps2D(dtype='fp16', scale=512.)

        iou1 = bbox_overlaps1(bboxes1.cuda(), bboxes2.cuda())
        iou2 = bbox_overlaps2(bboxes1.cuda(), bboxes2.cuda())
        iou3 = bbox_overlaps3(bboxes1.cuda(), bboxes2.cuda())
        iou4 = bbox_overlaps3(bboxes1, bboxes2)

        iou2 = (iou1.flatten() - iou2.flatten()) < 1e-6
        iou3 = (iou1.flatten() - iou3.flatten()) < 1e-6
        iou4 = (iou1.cpu().flatten() - iou4.flatten()) < 1e-6
        assert iou2.sum() == len(iou1.flatten())
        assert iou3.sum() == len(iou1.flatten())
        assert iou4.sum() == len(iou1.flatten())
