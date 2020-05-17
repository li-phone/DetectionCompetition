# -*- coding: utf-8 -*-
import os
import time
import json
import codecs
import numpy as np
from PIL import Image

# import tensorflow as tf
# from keras import backend as K
# from keras.layers import Input
# from collections import OrderedDict
# from yolo3.utils import letterbox_image
# from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body

try:
    import log

    logger = log.getLogger(__name__)
    from model_service.pytorch_model_service import PTServingBaseService
    from metric.metrics_manager import MetricsManager
except:
    print('model_service error!')

import torch
# from mmdet.apis import init_detector, inference_detector

cat2label = {
    1: {'name': '一次性快餐盒', 'id': 1, 'supercategory': '其他垃圾'},
    2: {'name': '书籍纸张', 'id': 2, 'supercategory': '可回收物'},
    3: {'name': '充电宝', 'id': 3, 'supercategory': '可回收物'},
    4: {'name': '剩饭剩菜', 'id': 4, 'supercategory': '厨余垃圾'},
    5: {'name': '包', 'id': 5, 'supercategory': '可回收物'},
    6: {'name': '垃圾桶', 'id': 6, 'supercategory': '可回收物'},
    7: {'name': '塑料器皿', 'id': 7, 'supercategory': '可回收物'},
    8: {'name': '塑料玩具', 'id': 8, 'supercategory': '可回收物'},
    9: {'name': '塑料衣架', 'id': 9, 'supercategory': '可回收物'},
    10: {'name': '大骨头', 'id': 10, 'supercategory': '厨余垃圾'},
    11: {'name': '干电池', 'id': 11, 'supercategory': '有害垃圾'},
    12: {'name': '快递纸袋', 'id': 12, 'supercategory': '可回收物'},
    13: {'name': '插头电线', 'id': 13, 'supercategory': '可回收物'},
    14: {'name': '旧衣服', 'id': 14, 'supercategory': '可回收物'},
    15: {'name': '易拉罐', 'id': 15, 'supercategory': '可回收物'},
    16: {'name': '枕头', 'id': 16, 'supercategory': '可回收物'},
    17: {'name': '果皮果肉', 'id': 17, 'supercategory': '厨余垃圾'},
    18: {'name': '毛绒玩具', 'id': 18, 'supercategory': '可回收物'},
    19: {'name': '污损塑料', 'id': 19, 'supercategory': '其他垃圾'},
    20: {'name': '污损用纸', 'id': 20, 'supercategory': '其他垃圾'},
    21: {'name': '洗护用品', 'id': 21, 'supercategory': '可回收物'},
    22: {'name': '烟蒂', 'id': 22, 'supercategory': '其他垃圾'},
    23: {'name': '牙签', 'id': 23, 'supercategory': '其他垃圾'},
    24: {'name': '玻璃器皿', 'id': 24, 'supercategory': '可回收物'},
    25: {'name': '砧板', 'id': 25, 'supercategory': '可回收物'},
    26: {'name': '筷子', 'id': 26, 'supercategory': '其他垃圾'},
    27: {'name': '纸盒纸箱', 'id': 27, 'supercategory': '可回收物'},
    28: {'name': '花盆', 'id': 28, 'supercategory': '其他垃圾'},
    29: {'name': '茶叶渣', 'id': 29, 'supercategory': '厨余垃圾'},
    30: {'name': '菜帮菜叶', 'id': 30, 'supercategory': '厨余垃圾'},
    31: {'name': '蛋壳', 'id': 31, 'supercategory': '厨余垃圾'},
    32: {'name': '调料瓶', 'id': 32, 'supercategory': '可回收物'},
    33: {'name': '软膏', 'id': 33, 'supercategory': '有害垃圾'},
    34: {'name': '过期药物', 'id': 34, 'supercategory': '有害垃圾'},
    35: {'name': '酒瓶', 'id': 35, 'supercategory': '可回收物'},
    36: {'name': '金属厨具', 'id': 36, 'supercategory': '可回收物'},
    37: {'name': '金属器皿', 'id': 37, 'supercategory': '可回收物'},
    38: {'name': '金属食品罐', 'id': 38, 'supercategory': '可回收物'},
    39: {'name': '锅', 'id': 39, 'supercategory': '可回收物'},
    40: {'name': '陶瓷器皿', 'id': 40, 'supercategory': '其他垃圾'},
    41: {'name': '鞋', 'id': 41, 'supercategory': '可回收物'},
    42: {'name': '食用油桶', 'id': 42, 'supercategory': '可回收物'},
    43: {'name': '饮料瓶', 'id': 43, 'supercategory': '可回收物'},
    44: {'name': '鱼骨', 'id': 44, 'supercategory': '厨余垃圾'}
}


class ObjectDetectionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('model_name:', model_name, ', model_path', model_path)
        if torch.cuda.is_available() is True:
            device = 'cuda:0'
            print('use tf GPU version,', torch.__version__)
        else:
            device = 'cpu'
            print('use tf CPU version,', torch.__version__)

        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(__file__), 'trained_weights_final.h5')
        self.input_image_key = 'images'

        self.cfg = 'model/configs/retinanet_r50_fpn_1x.py'
        self.checkpoint = 'model/work_dirs/retinanet_r50_fpn_1x/epoch_12.pth'
        self.cat2label = cat2label
        self.cfg_name = os.path.basename(self.cfg[:-3])

        print('cfg: ', self.cfg, 'checkpoint:', self.checkpoint)
        print('starting init detector model...')
        # self.infer_model = init_detector(self.cfg, self.checkpoint, device=device)
        print('load weights file success')

        # self.anchors_path = os.path.join(os.path.dirname(__file__), 'train_anchors.txt')
        # self.classes_path = os.path.join(os.path.dirname(__file__), 'train_classes.txt')
        # self.score = 0.3
        # self.iou = 0.45
        # self.model_image_size = (416, 416)
        #
        # self.label_map = parse_classify_rule(os.path.join(os.path.dirname(__file__), 'classify_rule.json'))
        # self.class_names = self._get_class()
        # self.anchors = self._get_anchors()
        # self.sess = K.get_session()
        # with self.sess.as_default():
        #     with self.sess.graph.as_default():
        #         self.K_learning_phase = K.learning_phase()
        #         self.boxes, self.scores, self.classes = self.generate()
        # print('load weights file success')

    def is_tf_gpu_version(self):
        from tensorflow.python.client import device_lib
        is_gpu_version = False
        devices_info = device_lib.list_local_devices()
        for device in devices_info:
            if 'GPU' == str(device.device_type):
                is_gpu_version = True
                break

        return is_gpu_version

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with codecs.open(classes_path, 'r', 'utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting

        self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
            if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match

        print('{} model, anchors, and classes loaded.'.format(model_path))

        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                preprocessed_data[k] = image
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        image = data[self.input_image_key]

        # import cv2 as cv
        # image = cv.imdecode(image.read(), np.uint8, 1)

        image = image.convert("RGB")
        image = np.array(image)

        results = dict(detection_classes=[], detection_scores=[], detection_boxes=[])
        # try:
        #     result = inference_detector(self.infer_model, image)
        #     for j, rows in enumerate(result):
        #         for r in rows:
        #             r = [float(_) for _ in r]
        #             label = self.cat2label[j + 1]['supercategory'] + '/' + self.cat2label[j + 1]['name']
        #             results['detection_classes'].append(label)
        #             results['detection_scores'].append(round(r[4], 4))
        #             bbox = [r[1], r[0], r[3], r[2]]
        #             bbox = [round(_, 1) for _ in bbox]
        #             results['detection_boxes'].append(bbox)
        # except:
        #     print('inference_detector error!')
        return results

        # for j, rows in enumerate(range(44)):
        #     rows = np.random.random(4) * 1000
        #     rows = np.sort(rows)
        #     rows = [np.append(rows, np.random.random())]
        #     for r in rows:
        #         r = [float(_) for _ in r]
        #         label = self.cat2label[j + 1]['supercategory'] + '/' + self.cat2label[j + 1]['name']
        #         results['detection_classes'].append(label)
        #         results['detection_scores'].append(round(r[4], 4))
        #         # bbox = [float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])]
        #         bbox = [r[1], r[0], r[3], r[2]]
        #         bbox = [round(_, 1) for _ in bbox]
        #         results['detection_boxes'].append(bbox)
        # return results

        # if self.model_image_size != (None, None):
        #     assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        #     boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # else:
        #     new_image_size = (image.width - (image.width % 32),
        #                       image.height - (image.height % 32))
        #     boxed_image = letterbox_image(image, new_image_size)
        # image_data = np.array(boxed_image, dtype='float32')
        #
        # # print(image_data.shape)
        # image_data /= 255.
        # image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        #
        # out_boxes, out_scores, out_classes = self.sess.run(
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [image.size[1], image.size[0]],
        #         self.K_learning_phase: 0
        #     })
        #
        # result = OrderedDict()
        # if out_boxes is not None:
        #     detection_class_names = []
        #     for class_id in out_classes:
        #         class_name = self.class_names[int(class_id)]
        #         class_name = self.label_map[class_name] + '/' + class_name
        #         detection_class_names.append(class_name)
        #     out_boxes_list = []
        #     for box in out_boxes:
        #         out_boxes_list.append(
        #             [round(float(v), 1) for v in box])  # v是np.float32类型，会导致无法json序列化，因此使用float(v)转为python内置float类型
        #     result['detection_classes'] = detection_class_names
        #     result['detection_scores'] = [round(float(v), 4) for v in out_scores]
        #     result['detection_boxes'] = out_boxes_list
        # else:
        #     result['detection_classes'] = []
        #     result['detection_scores'] = []
        #     result['detection_boxes'] = []
        # return result

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
        data = {'images': {'file_name': p} for i, p in enumerate(paths)}
        data = self._preprocess(data)
        data = self._inference(data)
        data = self._postprocess(data)
        print(data)


def parse_classify_rule(json_path=''):
    with codecs.open(json_path, 'r', 'utf-8') as f:
        rule = json.load(f)
    label_map = {}
    for super_label, labels in rule.items():
        for label in labels:
            label_map[label] = super_label
    return label_map


if __name__ == '__main__':
    data = {}
    ObjectDetectionService(None, None)._test_run()
