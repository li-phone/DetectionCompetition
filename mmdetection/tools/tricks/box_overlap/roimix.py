import mmcv
import copy
import numpy as np


# RoIMix: Proposal-Fusion among Multiple Images for Underwater Object Detection, Lin et al..
# arXiv: https://arxiv.org/abs/1911.03029
class RoiMix(object):

    def __init__(self, switch_prob=1., alpha=0.1, level='ground_truth', type='single', repeated=False,
                 distribution='beta', use_max=True, min_iou=None):
        self.switch_prob = switch_prob
        self.alpha = alpha
        self.level = level
        # To do
        # Proposal level RoI Mix
        self.type = type
        self.repeated = repeated
        self.distribution = distribution
        self.use_max = use_max
        self.min_iou = min_iou

    def _roimix(self, b1, b2, img):
        b1 = [int(_) for _ in b1]
        b2 = [int(_) for _ in b2]
        roi2 = img[b2[1]:b2[3], b2[0]:b2[2], :].copy()
        b1w, b1h = (b1[2] - b1[0]), (b1[3] - b1[1])
        if self.min_iou is not None:
            min_area = b1w * b1h * self.min_iou
            min_w, min_h = np.sqrt(min_area), np.sqrt(min_area)
            scale = [max(0., b1h - min_h), max(0., b1w - min_w)]
            scale[0] = b1h if scale[0] == 0 else scale[0]
            scale[1] = b1w if scale[1] == 0 else scale[1]
        else:
            scale = (b1h, b1w)
        roi2, scale_factor = mmcv.imrescale(roi2, tuple(scale), return_scale=True)
        lamb = np.random.beta(self.alpha, self.alpha)
        if self.use_max:
            new_lamb = max(lamb, 1 - lamb)
        else:
            new_lamb = lamb
        img[b1[1]:b1[3], b1[0]:b1[2], :] = new_lamb * img[b1[1]:b1[3], b1[0]:b1[2], :] + (
                1 - new_lamb) * roi2
        return img

    def roimix(self, img, _gt_boxes):
        gt_boxes = copy.deepcopy(_gt_boxes)
        img = img.astype(np.float) / 255.0
        if self.type == 'single':
            if self.repeated:
                for i, b1 in enumerate(gt_boxes):
                    idx = np.random.randint(len(gt_boxes))
                    b2 = gt_boxes[idx]
                    img = self._roimix(b1, b2, img)
            else:
                while len(gt_boxes) > 1:
                    idx = np.random.randint(len(gt_boxes))
                    b1 = gt_boxes[idx]
                    gt_boxes.pop(idx)
                    idx = np.random.randint(len(gt_boxes))
                    b2 = gt_boxes[idx]
                    gt_boxes.pop(idx)
                    img = self._roimix(b1, b2, img)
        else:
            raise Exception('Only support single type in ground_truth level in current.')

        new_img = np.array(img * 255).astype(np.uint8)
        return new_img

    def __call__(self, results):
        if 'roimix' not in results:
            use_roimix = True if np.random.rand() < self.switch_prob else False
            results['roimix'] = use_roimix
        if results['boxmixup']:
            results['img'] = self.roimix(results['img'], results['gt_boxes'])
        # cv_showimg(**results, old_flag=old_flag)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(level={}, type={})'.format(self.level, self.type)
        return repr_str
