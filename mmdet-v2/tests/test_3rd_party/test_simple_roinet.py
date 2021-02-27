import numpy as np
import requests
import cv2
import time
import torch
import os.path as osp
from mmdet.models.backbones.simple_roinet import SimpleROINet


class TestSimpleROINet(object):

    @classmethod
    def setup_class(cls):
        pass

    def test_simple_roinet(self):
        settings = dict()
        simple_roinet = SimpleROINet(**settings)
        start = time.time()
        result = simple_roinet(torch.rand((1, 3, 320, 320)))
        end = time.time()
        # times: 86.52158761024475s
        print('times: {}s'.format(end - start))
        out = result.shape
        assert int(out[0]) == 1 and int(out[1]) == 1 and int(out[2]) == 10 and int(out[3]) == 10
