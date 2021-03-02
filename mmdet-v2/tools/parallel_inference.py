import os
import time
import json
import cv2
import torch
import numpy as np
from mmcv.ops.nms import batched_nms
from pycocotools.coco import COCO
from mmdet.datasets.pipelines import Compose
from mmdet.third_party.parallel import Parallel
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


class Config(object):
    # process module
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        # dict(type='SliceROI', training=False, padding=10),
        dict(type='SliceImage', training=False, window=(1000, 1000), step=(500, 500), order_index=False,
             is_keep_none=True),
    ]
    compose = Compose(train_pipeline)

    # data module
    img_dir = "/home/lifeng/undone-work/DefectNet/tools/data/tile/raw/tile_round1_testB_20210128/testB_imgs"
    test_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/submit_testB.json"
    save_file = "/home/lifeng/undone-work/DetCompetition/mmdet-v2/work_dirs/tile/cascade_rcnn_x101_64x4d_fpn_20e_cut_800x800/resize_4000x3000_testB.json"
    original_coco = COCO(test_file)
    label2name = {x['id']: x['name'] for x in original_coco.dataset['categories']}

    # inference module
    device = 'cuda:1'
    config_file = '../configs/tile/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_cut_800x800.py'
    checkpoint_file = 'work_dirs/tile/cascade_rcnn_x101_64x4d_fpn_20e_cut_800x800/epoch_20.pth'
    model = init_detector(config_file, checkpoint_file, device=device)


def process(image, **kwargs):
    save_results = dict(result=[])
    config = kwargs['config']
    image['filename'] = image['file_name']
    results = {'img_prefix': config.img_dir, 'img_info': image}
    results = config.compose(results)
    if results is None: return save_results
    if isinstance(results, dict):
        results = [results]
    bboxes = np.empty([0, 4], dtype=np.float32)
    scores = np.empty([0], dtype=np.float32)
    labels = np.empty([0], dtype=np.int)
    for i, result in enumerate(results):
        bbox_result = inference_detector(config.model, result['img'])
        for j in range(len(bbox_result)):
            if len(bbox_result[j]) <= 0:
                continue
            if 'slice_roi__left_top' in result:
                bbox_result[j][:, 0] += result['slice_roi__left_top'][0]
                bbox_result[j][:, 1] += result['slice_roi__left_top'][1]
                bbox_result[j][:, 2] += result['slice_roi__left_top'][0]
                bbox_result[j][:, 3] += result['slice_roi__left_top'][1]
            if 'slice_image__left_top' in result:
                bbox_result[j][:, 0] += result['slice_image__left_top'][0]
                bbox_result[j][:, 1] += result['slice_image__left_top'][1]
                bbox_result[j][:, 2] += result['slice_image__left_top'][0]
                bbox_result[j][:, 3] += result['slice_image__left_top'][1]
            bboxes = np.append(bboxes, bbox_result[j][:, :4], axis=0)
            scores = np.append(scores, bbox_result[j][:, 4], axis=0)
            labels = np.append(labels, [j + 1] * len(bbox_result[j]), axis=0)
    if len(bboxes) < 1 or len(scores) < 1 or len(labels) < 1:
        return save_results
    bboxes = torch.from_numpy(bboxes)
    scores = torch.from_numpy(scores)
    labels = torch.from_numpy(labels)
    bboxes, keep = batched_nms(bboxes, scores, labels, nms_cfg=config.model.cfg.test_cfg.rcnn.nms)
    labels = labels[keep]
    for r, label in zip(bboxes, labels):
        bbox = list(map(float, r[:4]))
        if int(label) not in config.label2name:
            continue
        category_id, score = config.label2name[int(label)], r[4]
        save_results['result'].append({'name': str(image['filename']), 'category': int(category_id),
                                       'bbox': bbox, 'score': float(score)})
    # from pandas.io.json import json_normalize
    # from mmcv.visualization.image import imshow_det_bboxes
    # df = json_normalize(save_results['result'])
    # anns = np.array(df['bbox'])
    # score = np.array(df['score'])
    # bbox = np.array([[b[0], b[1], b[2], b[3], score[i]] for i, b in enumerate(anns)])
    # img = imshow_det_bboxes(os.path.join(config.img_dir, image['file_name']), bbox, labels, show=False)
    # cv2.imwrite(image['file_name'], img)
    end = time.time()
    print('s/img: {:.2f}'.format(end - kwargs['time']['start']))
    kwargs['time']['start'] = end
    return save_results


def parallel_infer():
    config = Config()
    if not os.path.exists(os.path.dirname(config.save_file)):
        os.makedirs(os.path.dirname(config.save_file))
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(tasks=config.original_coco.dataset['images'],
                    process=process, collect=['result'], workers_num=1,
                    process_params=process_params, print_process=10)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {}s'.format(end - start))
    with open(config.save_file, "w") as fp:
        json.dump(results['result'], fp, indent=4, ensure_ascii=False)
    print("process ok!")


if __name__ == '__main__':
    parallel_infer()
