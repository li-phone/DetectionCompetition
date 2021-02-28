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
        dict(type='SliceImage', training=True, window=(800, 800), step=(400, 400), order_index=False)
    ]
    compose = Compose(train_pipeline)

    # data module
    img_dir = "/home/lifeng/undone-work/dataset/detection/tile/raw/tile_round1_train_20201231/train_imgs/"
    save_img_dir = "/home/lifeng/undone-work/dataset/detection/tile/trainval/cut_800x800/"
    ann_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/instance_all-check.json"
    save_ann_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/cut_800x800/cut_800x800_all.json"
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
        tmp_image = {k: v for k, v in image.items()}
        x1, y1, x2, y2 = result['slice_image']['left_top']
        tmp_image['file_name'] = "{}__{:04d}_{:04d}_{:04d}_{:04d}.jpg".format(tmp_image['file_name'], x1, y1, x2, y2)
        tmp_image['height'] = result['img'].shape[0]
        tmp_image['width'] = result['img'].shape[1]
        save_name = os.path.join(config.save_img_dir, tmp_image['file_name'])
        if not os.path.exists(save_name):
            cv2.imwrite(save_name, result['img'])
        tmp_image['id'] = tmp_image['file_name']
        kwargs['__results__']['images'].append(tmp_image)
        for bbox, label in zip(result['gt_bboxes'], result['gt_labels']):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
            ann = dict(
                id=len(kwargs['__results__']['annotations']),
                image_id=tmp_image['id'],
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
    return save_results


def parallel_slice():
    config = Config()
    if not os.path.exists(config.save_img_dir):
        os.makedirs(config.save_img_dir)
    if not os.path.exists(os.path.dirname(config.save_ann_file)):
        os.makedirs(os.path.dirname(config.save_ann_file))
    process_params = dict(config=config)
    settings = dict(tasks=config.original_coco.dataset['images'],
                    process=process, collect=['images', 'annotations'], workers_num=10,
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
