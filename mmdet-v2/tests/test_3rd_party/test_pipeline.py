import numpy as np
import pytest
import requests
import cv2
import time
import torch
import os.path as osp
from mmdet.datasets.pipelines import Compose


class TestPipeline(object):

    @classmethod
    def setup_class(cls):
        cls.img_prefix = '../data/'
        cls.img_info = {'filename': 'color.jpg'}

    def test_random_flip_diagonal(self):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='RandomFlip', flip_ratio=1., direction='diagonal'),
        ]
        compose = Compose(pipeline)
        results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
        cv2.imwrite(self.img_prefix + 'test_random_flip_diagonal.jpg', results['img'])

    def test_random_flip_horizontal(self):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='RandomFlip', flip_ratio=1., direction='horizontal'),
        ]
        compose = Compose(pipeline)
        results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
        cv2.imwrite(self.img_prefix + 'test_random_flip_horizontal.jpg', results['img'])

    def test_random_flip_vertical(self):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='RandomFlip', flip_ratio=1., direction='vertical'),
        ]
        compose = Compose(pipeline)
        results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
        cv2.imwrite(self.img_prefix + 'test_random_flip_vertical.jpg', results['img'])

    def test_random_shift(self):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='RandomShift'),
        ]
        compose = Compose(pipeline)
        results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
        cv2.imwrite(self.img_prefix + 'test_random_shift.jpg', results['img'])

    def test_random_auto_augment(self):
        policies1 = [
            [dict(type='RandomFlip', flip_ratio=0.0, direction='horizontal')],
            [dict(type='RandomFlip', flip_ratio=1.0, direction='horizontal')],
            [dict(type='RandomFlip', flip_ratio=1.0, direction='vertical')],
            [dict(type='RandomFlip', flip_ratio=1.0, direction='diagonal')],
        ]
        policies2 = [
            [dict(type='Rotate', level=10, prob=1.0, max_rotate_angle=0)],
            [dict(type='Rotate', level=10, prob=1.0, max_rotate_angle=90)],
            [dict(type='Rotate', level=10, prob=1.0, max_rotate_angle=180)],
            [dict(type='Rotate', level=10, prob=1.0, max_rotate_angle=270)],
        ]
        for p1 in policies1:
            for p2 in policies2:
                pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(type='AutoAugment', policies=[p1]),
                    dict(type='AutoAugment', policies=[p2]),
                ]
                compose = Compose(pipeline)
                results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
                cv2.imwrite(
                    self.img_prefix + f'test_random_auto_augment_{p1[0]["direction"]}_{p2[0]["max_rotate_angle"]}.jpg',
                    results['img'])

    def test_random_rotate(self):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Rotate', level=5, prob=1),
        ]
        compose = Compose(pipeline)
        results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
        cv2.imwrite(self.img_prefix + 'test_random_rotate.jpg', results['img'])

    def test_random_rotate90(self):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Rotate', level=10, prob=1, max_rotate_angle=270),
        ]
        compose = Compose(pipeline)
        results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
        cv2.imwrite(self.img_prefix + 'test_random_rotate90.jpg', results['img'])

    def test_cutout(self):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='CutOut', n_holes=2, cutout_ratio=[(0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6)]),
        ]
        compose = Compose(pipeline)
        results = compose({'img_prefix': self.img_prefix, 'img_info': self.img_info})
        cv2.imwrite(self.img_prefix + 'test_cutout.jpg', results['img'])

    def test_concat(self):
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), ratio_range=(0.8, 1.2), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Concat'),
        ]
        compose = Compose(train_pipeline)
        multi_results = [
            {'img_prefix': self.img_prefix, 'img_info': self.img_info, 'bbox_fields': [],
             'ann_info': {'bboxes': np.array([200, 210, 240, 250]), 'labels': np.array([1])}},
            {'img_prefix': self.img_prefix, 'img_info': self.img_info, 'bbox_fields': [],
             'ann_info': {'bboxes': np.array([300, 400, 340, 450]), 'labels': np.array([1])}},
        ]
        results = compose(multi_results)
        cv2.imwrite(self.img_prefix + 'test_concat.jpg', results['img'])
