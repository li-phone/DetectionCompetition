# -*- coding: utf-8 -*-
import os
import sys
import pip

# os.system('nvcc -V')
# os.system('python -V')
# os.system('gcc --version')
# os.system('nvidia-smi')
# os.system('uname')
# print('current dir')
# os.system('pwd')
# os.system('ls')
# os.system('ls model')
# os.system('pip install model/whl/Cython-0.29.17-cp36-cp36m-manylinux1_x86_64.whl')
# os.system('pip install -r requirements.txt')
# os.system('pip install model/whl/mmdet-2.0.0+2d8bfea-cp37-cp37m-linux_x86_64.whl')

# os.system('pwd')
# print('os.environ:')
# for k, v in os.environ.items():
#     print('{}: {}'.format(k, v))
# print('sys.path:')
# for i, v in enumerate(sys.path):
#     print('{}: {}'.format(i, v))


import time
import json
import codecs
import numpy as np
from PIL import Image
from collections import OrderedDict

# modelarts import
import torch
import mmdet

print('import mmdet ok!')

import log

logger = log.getLogger(__name__)
from metric.metrics_manager import MetricsManager
from model_service.pytorch_model_service import PTServingBaseService

import config

try:
    from mmdet.apis import init_detector, inference_detector
except:
    print('from mmdet.apis import init_detector, inference_detector error!')


class ObjectDetectionService(PTServingBaseService):
    def __init__(self, cfg=None, model_path=None):
        if torch.cuda.is_available() is True:
            print('use torch GPU version,', torch.__version__)
        else:
            print('use torch CPU version,', torch.__version__)
        if cfg is None:
            self.cfg = config.cfg
        else:
            self.cfg = cfg
        if model_path is None:
            self.model_path = config.model_path
        else:
            self.model_path = model_path
        self.cat2label = config.cat2label
        self.model_name = os.path.basename(self.cfg[:-3])
        # self.model = init_detector(self.cfg, self.model_path, device='cuda:0')

        print('load weights file success')

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                image = np.array(image)
                preprocessed_data[k] = image
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        images = data

        results = dict(detection_classes=[], detection_scores=[], detection_boxes=[])
        for img_id, image in images.items():
            j = np.random.randint(40)
            r = [np.random.randint(10, 500), np.random.randint(10, 500),
                 np.random.randint(10, 500), np.random.randint(10, 500),
                 1 - np.random.random()]
            label = self.cat2label[j + 1]['supercategory'] + '/' + self.cat2label[j + 1]['name']
            results['detection_classes'].append(label)
            results['detection_scores'].append(r[4])
            bbox = [float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])]
            bbox = [round(_, 2) for _ in bbox]
            results['detection_boxes'].append(bbox)
            pass
            # result = inference_detector(self.model, image)
            # for j, rows in enumerate(result):
            #     for r in rows:
            #         label = self.cat2label[j + 1]['supercategory'] + '/' + self.cat2label[j + 1]['name']
            #         results['detection_classes'].append(label)
            #         results['detection_scores'].append(r[4])
            #         bbox = [float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])]
            #         bbox = [round(_, 2) for _ in bbox]
            #         results['detection_boxes'].append(bbox)
        return results

    def _postprocess(self, data):
        return data

    # def inference(self, data):
    #     '''
    #     Wrapper function to run preprocess, inference and postprocess functions.
    #
    #     Parameters
    #     ----------
    #     data : map of object
    #         Raw input from request.
    #
    #     Returns
    #     -------
    #     list of outputs to be sent back to client.
    #         data to be sent back
    #     '''
    #     pre_start_time = time.time()
    #     data = self._preprocess(data)
    #     infer_start_time = time.time()
    #     # Update preprocess latency metric
    #     pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
    #     logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
    #
    #     if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
    #         MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
    #
    #     data = self._inference(data)
    #     infer_end_time = time.time()
    #     infer_in_ms = (infer_end_time - infer_start_time) * 1000
    #
    #     logger.info('infer time: ' + str(infer_in_ms) + 'ms')
    #     data = self._postprocess(data)
    #
    #     # Update inference latency metric
    #     post_time_in_ms = (time.time() - infer_end_time) * 1000
    #     logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
    #     if self.model_name + '_LatencyInference' in MetricsManager.metrics:
    #         MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
    #
    #     # Update overall latency metric
    #     if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
    #         MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
    #
    #     logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
    #     data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
    #     return data

    def _test_run(self):
        data = {
            '0': {
                'file_name': r'E:\liphone\data\images\detections\garbage_huawei\JPEGImages\5d54a2d0231f224b592725f9fef0d90.jpg',
            },
            '1': {
                'file_name': r'E:\liphone\data\images\detections\garbage_huawei\JPEGImages\5d77f426ea2b5a13e427b0747631e6d.jpg',
            },
        }
        data = self._preprocess(data)
        data = self._inference(data)
        data = self._postprocess(data)
        print(data)
        pass


def parse_classify_rule(json_path=''):
    with codecs.open(json_path, 'r', 'utf-8') as f:
        rule = json.load(f)
    label_map = {}
    for super_label, labels in rule.items():
        for label in labels:
            label_map[label] = super_label
    return label_map


if __name__ == '__main__':
    ObjectDetectionService()._test_run()
