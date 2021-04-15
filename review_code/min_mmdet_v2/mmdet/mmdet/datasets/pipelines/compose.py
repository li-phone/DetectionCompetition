import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.

            我们修改了compose使得它能够支持多个data的处理
            返回：
                返回单个data或者若干个data [list]
        """
        if 'img' in data and isinstance(data['img'], list):
            many_data = dict(img_metas=[], img=[])
            for img in data['img']:
                data_ = dict(img=img)
                for t in self.transforms:
                    data_ = t(data_)
                    if data_ is None:
                        return None
                many_data['img'].extend(data_['img'])
                many_data['img_metas'].extend(data_['img_metas'])
            return many_data

        last_ind = 0
        for i, t in enumerate(self.transforms):
            last_ind = i
            data = t(data)
            if isinstance(data, list):
                break
            if data is None:
                return None
        if (last_ind + 1) < len(self.transforms) and isinstance(data, list):
            data, data_list = [], data
            for idx, d in enumerate(data_list):
                for i, t in enumerate(self.transforms[last_ind + 1:]):
                    d = t(d)
                    if d is None:
                        break
                data.append(d)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
