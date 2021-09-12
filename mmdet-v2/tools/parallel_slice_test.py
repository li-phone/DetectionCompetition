import os
import json
import glob
import cv2
import time
import numpy as np
from pycocotools.coco import COCO
from mmdet.datasets.pipelines import Compose
from mmdet.third_party.parallel import Parallel
from x2coco import _get_box


class Config(object):
    # process module
    # 2000 x 2000
    train_pipeline1 = [
        dict(type='LoadImageFromFile'),
        dict(type='SliceImage', overlap=1, base_win=(1000, 1000), step=(0.5, 0.5), resize=(0.5, 0.5),
             keep_none=True, is_testing=False),
    ]
    train_pipeline2 = [
        dict(type='LoadImageFromFile'),
        dict(type='SliceImage', overlap=1, base_win=(1000, 1000), step=(0.5, 0.5), resize=(1.0, 1.0),
             keep_none=True, is_testing=False),
    ]
    train_pipeline3 = [
        dict(type='LoadImageFromFile'),
        dict(type='SliceImage', overlap=1, base_win=(1000, 1000), step=(0.5, 0.5), resize=(1.5, 1.5),
             keep_none=True, is_testing=False),
    ]
    composes = [Compose(train_pipeline1), Compose(train_pipeline2), Compose(train_pipeline3)]

    # data module
    img_dir = "data/orange2/test_A/images/"
    tasks = glob.glob(img_dir + "*")
    save_img_dir = "data/orange2/test_A-slice_1000x1000-resize_3/"


def plt_show():
    # from pandas.io.json import json_normalize
    # from mmcv.visualization.image import imshow_bboxes
    # anns = json_normalize(kwargs['__results__']['annotations'])
    # for id in np.unique(anns['image_id']):
    #     bbox = anns[anns['image_id'] == id]['bbox']
    #     bbox = [b for b in bbox]
    #     bbox = np.array(bbox)
    #     bbox[:, 2] += bbox[:, 0]
    #     bbox[:, 3] += bbox[:, 1]
    #     img_filename = os.path.join(config.save_img_dir, id)
    #     img = imshow_bboxes(img_filename, bbox, show=False)
    #     cv2.imwrite(id, img)
    pass


def process(image, **kwargs):
    save_results = {}
    config = kwargs['config']
    # 多尺度切割
    for compose in config.composes:
        results = {'img_prefix': config.img_dir, 'img_info': {'filename': os.path.basename(image)}}
        results = compose(results)
        if results is None: return save_results
        if not isinstance(results, (list, tuple)): results = [results]
        for i, result in enumerate(results):
            tmp_img = {'file_name': image}
            if 'slice_image' in result:
                x1, y1, x2, y2 = result['slice_image']['window']
                tmp_img['file_name'] = "{}__fx_{:.3f}_fy_{:.3f}__{:06d}_{:06d}_{:06d}_{:06d}.jpg".format(
                    tmp_img['file_name'][:-4],
                    compose.transforms[1].resize[0],
                    compose.transforms[1].resize[1],
                    x1, y1, x2, y2, )
                tmp_img['height'] = result['img'].shape[0]
                tmp_img['width'] = result['img'].shape[1]
                tmp_img['id'] = tmp_img['file_name']
            save_name = os.path.join(config.save_img_dir, tmp_img['file_name'])
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            if True or not os.path.exists(save_name):
                # print(save_name)
                cv2.imwrite(save_name, result['img'])
    end = time.time()
    print('second/img: {:.2f}'.format(end - kwargs['time']['start']))
    kwargs['time']['start'] = end
    return save_results


def parallel_slice():
    config = Config()
    if not os.path.exists(config.save_img_dir):
        os.makedirs(config.save_img_dir)
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(tasks=config.tasks, process=process, collect=[], workers_num=8,
                    process_params=process_params, print_process=10)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {}s'.format(end - start))
    print("process ok!")


if __name__ == '__main__':
    parallel_slice()
