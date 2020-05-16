# -*- coding: utf-8 -*-
import os
import time
import json
import codecs
import numpy as np
from PIL import Image
from collections import OrderedDict

try:
    import log

    logger = log.getLogger(__name__)
    from metric.metrics_manager import MetricsManager
    from model_service.pytorch_model_service import PTServingBaseService
except:
    print('model_service error!')

# modelarts import
import torch

print('torch.__version__', torch.__version__)
from mmdet.apis import init_detector, inference_detector
import config


class ObjectDetectionService(PTServingBaseService):
    def __init__(self, cfg=None, model_path=None):
        if torch.cuda.is_available() is True:
            device = 'cuda:0'
            print('use torch GPU version,', torch.__version__)
        else:
            device = 'cpu'
            print('use torch CPU version,', torch.__version__)
        print('cfg:', cfg, ', model_path', model_path)
        self.cfg = config.cfg
        self.model_path = config.model_path
        self.cat2label = config.cat2label
        self.model_name = os.path.basename(self.cfg[:-3])
        print('starting init detector model...')
        print('cfg: ', self.cfg, 'model_path:', self.model_path)
        self.model = init_detector(self.cfg, self.model_path, device=device)
        print('load weights file success')

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                # image = Image.open(file_content)
                # image = np.array(image)
                preprocessed_data[k] = file_content
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        images = data

        results = dict(detection_classes=[], detection_scores=[], detection_boxes=[])
        for j, rows in enumerate(range(44)):
            rows = np.random.random(4) * 1000
            rows = np.sort(rows)
            rows = [np.append(rows, np.random.random())]
            for r in rows:
                r = [float(_) for _ in r]
                label = self.cat2label[j + 1]['supercategory'] + '/' + self.cat2label[j + 1]['name']
                results['detection_classes'].append(label)
                results['detection_scores'].append(round(r[4], 4))
                # bbox = [float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])]
                bbox = [r[1], r[0], r[3], r[2]]
                bbox = [round(_, 1) for _ in bbox]
                results['detection_boxes'].append(bbox)
        # # import cv2 as cv
        # for img_id, file_content in images.items():
        #     try:
        #         image = Image.open(file_content)
        #         image = np.array(image)
        #         result = inference_detector(self.model, image)
        #         for j, rows in enumerate(result):
        #             for r in rows:
        #                 r = [float(_) for _ in r]
        #                 label = self.cat2label[j + 1]['supercategory'] + '/' + self.cat2label[j + 1]['name']
        #                 results['detection_classes'].append(label)
        #                 results['detection_scores'].append(round(r[4], 4))
        #                 # bbox = [float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])]
        #                 bbox = [r[1], r[0], r[3], r[2]]
        #                 bbox = [round(_, 1) for _ in bbox]
        #                 results['detection_boxes'].append(bbox)
        #                 # pt1 = (int(bbox[0]), int(bbox[1]))
        #                 # pt2 = (int(bbox[2]), int(bbox[3]))
        #                 # cv.rectangle(image, pt1, pt2, color=(0, 0, 255))
        #         # cv.imshow('', image)
        #         # cv.waitKey()
        #     except:
        #         print('inference_detector error!')
        return results

    def _postprocess(self, data):
        return data

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object

            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data

    def _test_run(self):
        from glob import glob
        paths = glob('/home/liphone/undone-work/data/detection/garbage_huawei/images/*')
        data = {str(i): {'file_name': p} for i, p in enumerate(paths)}
        data = self._preprocess(data)
        data = self._inference(data)
        data = self._postprocess(data)
        print(data)


if __name__ == '__main__':
    ObjectDetectionService()._test_run()
