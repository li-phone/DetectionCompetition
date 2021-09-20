import os
import json
import cv2
import time
import numpy as np
from pycocotools.coco import COCO
from mmdet.datasets.pipelines import Compose
from mmdet.third_party.parallel import Parallel
from x2coco import _get_box


class Config(object):
    # process module
    # 3    tain数据切割，1.0 - 0.5 - 1.5，窗口 1000，gapratio = 0.5
    train_pipeline1 = [
        dict(type='LoadImageFromFile'),
        dict(type='SliceImage', overlap=1, base_win=(1000, 1000), step=(0.5, 0.5), resize=(0.5, 0.5), keep_none=False,
             is_testing=False),
    ]
    train_pipeline2 = [
        dict(type='LoadImageFromFile'),
        dict(type='SliceImage', overlap=1, base_win=(1000, 1000), step=(0.5, 0.5), resize=(1.0, 1.0), keep_none=False,
             is_testing=False),
    ]
    train_pipeline3 = [
        dict(type='LoadImageFromFile'),
        dict(type='SliceImage', overlap=1, base_win=(1000, 1000), step=(0.5, 0.5), resize=(1.5, 1.5), keep_none=False,
             is_testing=False),
    ]
    # composes = [Compose(train_pipeline2)]
    composes = [Compose(train_pipeline1), Compose(train_pipeline2), Compose(train_pipeline3)]

    # data module
    img_dir = "data/orange2/train/images"
    ann_file = "data/orange2/annotations/instance-train-checked.json"
    save_img_dir = "data/orange2/slice_1000x1000-overlap_0.5-resize_0.5_1.0_1.5/"
    save_ann_file = "data/orange2/annotations/slice_1000x1000-overlap_0.5-resize_0.5_1.0_1.5-train.json"
    original_coco = COCO(ann_file)


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
    save_results = dict(images=[], annotations=[])
    config = kwargs['config']
    image['filename'] = image['file_name']
    # 多尺度切割
    for compose in config.composes:
        annIds = config.original_coco.getAnnIds(imgIds=[image['id']])
        anns = config.original_coco.loadAnns(annIds)
        bboxes = np.array([x['bbox'] for x in anns])
        bboxes[:, 2:] += bboxes[:, :2]
        labels = np.array([x['category_id'] for x in anns])
        if 'bbox_uuid' in anns[0]:
            bbox_uuids = np.array([x['bbox_uuid'] for x in anns])
        else:
            bbox_uuids = np.array([x['id'] for x in anns])
        if 'group_uuid' in anns[0]:
            group_uuids = np.array([x['group_uuid'] for x in anns])
        else:
            group_uuids = np.array([x['image_id'] for x in anns])
        results = {
            'img_prefix': config.img_dir, 'img_info': image,
            'ann_info': {'bboxes': bboxes, 'labels': labels, 'bbox_uuid': bbox_uuids, 'group_uuid': group_uuids}}
        results = compose(results)
        if results is None: return save_results
        if not isinstance(results, (list, tuple)): results = [results]
        for i, result in enumerate(results):
            # 过滤小尺寸
            if compose.transforms[1].resize[0] == 0.5:
                keep_bboxes = result['ann_info']['bboxes']
                width = keep_bboxes[:, 2] - keep_bboxes[:, 0]
                height = keep_bboxes[:, 3] - keep_bboxes[:, 1]
                index1 = width >= 20
                index2 = height >= 20
                keep_idx = index1 + index2
                result['ann_info']['bboxes'] = result['ann_info']['bboxes'][keep_idx]
                result['ann_info']['labels'] = result['ann_info']['labels'][keep_idx]
                result['gt_bboxes'] = result['ann_info']['bboxes']
                result['gt_labels'] = result['ann_info']['labels']

            tmp_img = {k: v for k, v in image.items()}
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
            kwargs['__results__']['images'].append(tmp_img)
            for idx, bbox, label in zip(range(len(result['gt_bboxes'])), result['gt_bboxes'], result['gt_labels']):
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                ann = dict(
                    id=len(kwargs['__results__']['annotations']),
                    image_id=tmp_img['id'],
                    category_id=int(label),
                    bbox=_get_box(points),
                    iscrowd=0,
                    ignore=0,
                    area=area
                )
                ann['bbox_uuid'] = str(result['ann_info']['bbox_uuid'][idx])
                ann['group_uuid'] = str(result['ann_info']['group_uuid'][idx])
                kwargs['__results__']['annotations'].append(ann)
    end = time.time()
    print('second/img: {:.2f}'.format(end - kwargs['time']['start']))
    kwargs['time']['start'] = end
    return save_results


def parallel_slice():
    config = Config()
    if not os.path.exists(config.save_img_dir):
        os.makedirs(config.save_img_dir)
    if not os.path.exists(os.path.dirname(config.save_ann_file)):
        os.makedirs(os.path.dirname(config.save_ann_file))
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(tasks=config.original_coco.dataset['images'],
                    process=process, collect=['images', 'annotations'], workers_num=8,
                    process_params=process_params, print_process=10)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {}s'.format(end - start))
    dataset = config.original_coco.dataset
    tmp_coco = dict(info=dataset['info'], license=dataset['license'], categories=dataset['categories'],
                    images=results['images'], annotations=results['annotations'])
    with open(config.save_ann_file, "w") as fp:
        json.dump(tmp_coco, fp)
    print("process ok!")


if __name__ == '__main__':
    parallel_slice()
