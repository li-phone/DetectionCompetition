import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES

from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os.path as osp
import cv2


@PIPELINES.register_module()
class Compose(object):

    def __init__(self, transforms, ann_file=None, img_prefix=None,
                 type=None, mix_ratio=0.5, img_label=None, use_max=True, alpha=10, **kwargs):
        assert isinstance(transforms, collections.abc.Sequence)
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.type = type
        self.mix_ratio = mix_ratio
        self.img_label = img_label
        self.use_max = use_max
        self.alpha = alpha
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
        if self.type is not None:
            self.coco = COCO(self.ann_file)
            if 'img_label' in self.coco.dataset['images'][0] and self.img_label is not None:
                if isinstance(self.img_label, int):
                    self.img_label = [self.img_label]
                self.images = [r for r in self.coco.dataset['images'] if r['img_label'] in self.img_label]
            else:
                self.images = self.coco.dataset['images']

    def multi_mix(self, img1, img2):
        img1 = img1.astype(np.float) / 255.0
        img2 = img2.astype(np.float) / 255.0
        scale = (img1.shape[1], img1.shape[0],)
        img2 = cv2.resize(img2, scale, interpolation=cv2.INTER_LINEAR)
        # roi2, _rescale = mmcv.imrescale(roi2, tuple(scale), return_scale=True)
        lamb = np.random.beta(self.alpha, self.alpha)
        if self.use_max:
            new_lamb = max(lamb, 1 - lamb)
        else:
            new_lamb = lamb
        img1 = new_lamb * img1 + (1 - new_lamb) * img2
        img1 = np.array(img1 * 255).astype(np.uint8)
        return img1

    def __call__(self, data):
        if self.type is not None:
            if 'multiMix' not in data:
                flip = True if np.random.rand() <= self.mix_ratio else False
                data['multiMix'] = flip
            if data['multiMix']:
                ind = np.random.randint(len(self.images))
                image = self.images[ind]
                img2 = Image.open(osp.join(self.img_prefix, image['file_name'])).convert('RGB')
                img2 = np.array(img2)
        for i, t in enumerate(self.transforms):
            data = t(data)
            if data is None:
                return None
            if self.type is not None and i == 1 and data['multiMix']:
                img = self.multi_mix(data['img'], img2)
                data['img'] = img
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
