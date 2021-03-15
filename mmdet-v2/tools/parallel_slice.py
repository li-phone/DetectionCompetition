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
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        # dict(type='SliceROI', training=False),
        dict(type='SliceImage', overlap=0.7, training=True, window=(4000, 4000), step=(3500, 3500), order_index=False)
    ]
    compose = Compose(train_pipeline)

    # data module
    img_dir = "/home/lifeng/data/detection/track/panda_round1_train_202104_part1"
    ann_file = "/home/lifeng/data/detection/track/annotations/ori_instance_all_category.json"
    save_img_dir = "/home/lifeng/data/detection/track/trainval/overlap_70_all_category/"
    save_ann_file = "/home/lifeng/data/detection/track/annotations/overlap_70_all_category/instance_overlap_70_all_category.json"
    original_coco = COCO(ann_file)


def process(image, **kwargs):
    save_results = dict(images=[], annotations=[])
    config = kwargs['config']
    image['filename'] = image['file_name']
    annIds = config.original_coco.getAnnIds(imgIds=image['id'])
    anns = config.original_coco.loadAnns(annIds)
    bboxes = np.array([x['bbox'] for x in anns])
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    labels = np.array([x['category_id'] for x in anns])
    anns2 = {'bboxes': bboxes, 'labels': labels}
    results = {
        'img_prefix': config.img_dir,
        'img_info': image, 'ann_info': anns2}
    results = config.compose(results)
    if results is None: return save_results
    for i, result in enumerate(results):
        tmp_img = {k: v for k, v in image.items()}
        x1, y1, x2, y2 = result['slice_image']['left_top']
        tmp_img['file_name'] = "{}__{:06d}_{:06d}_{:06d}_{:06d}.jpg".format(tmp_img['file_name'][:-4], x1, y1, x2, y2)
        tmp_img['height'] = result['img'].shape[0]
        tmp_img['width'] = result['img'].shape[1]
        save_name = os.path.join(config.save_img_dir, tmp_img['file_name'])
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        if not os.path.exists(save_name):
            cv2.imwrite(save_name, result['img'])
        tmp_img['id'] = tmp_img['file_name']
        kwargs['__results__']['images'].append(tmp_img)
        for bbox, label in zip(result['gt_bboxes'], result['gt_labels']):
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
            kwargs['__results__']['annotations'].append(ann)
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
                    process=process, collect=['images', 'annotations'], workers_num=4,
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
