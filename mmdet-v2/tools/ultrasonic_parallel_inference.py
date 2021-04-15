import os
import time
import json
import cv2
import torch
import glob
import numpy as np
from pandas import json_normalize
from pandas.io.json import json_normalize
from mmcv.ops.nms import batched_nms
from mmdet.datasets.pipelines import Compose
from mmdet.third_party.parallel import Parallel
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


class Config(object):
    # data module
    img_dir = "data/ultrasonic/test_A/image_A/"
    # test_file = "data/underwater/panda_round1_test_B_annos_20210222/person_bbox_test_B.json"
    save_file = "work_dirs/ultrasonic/best-r50-mst_slice-ultrasonic.csv"
    tasks = glob.glob(img_dir + "*.bmp")
    slice_num = 10000
    nms_whole = False
    # label2name = {x['id']: x['name'] for x in original_coco.dataset['categories']}
    label2name = {0: 'ball', 1: 'circle cage', 2: 'cube', 3: 'cylinder', 4: 'human body', 5: 'metal bucket', 6: 'square cage', 7: 'tyre'}

    # inference module
    device = 'cuda:0'
    config_file = '../configs/ultrasonic/best-r50-mst_slice-ultrasonic.py'
    checkpoint_file = 'work_dirs/best-r50-mst_slice-ultrasonic/latest.pth'
    model = init_detector(config_file, checkpoint_file, device=device)


def mkdirs(path, is_file=True):
    if is_file:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def plt_bbox(img, boxes, labels, idx=None):
    # show test image
    import matplotlib.pyplot as plt
    from mmcv.visualization.image import imshow_det_bboxes
    img = np.array(img)
    img = imshow_det_bboxes(img, boxes, labels, show=False)
    # plt.imshow(img)
    file_name = "__show_img__/" + '{:04d}'.format(idx) + '.jpg'
    mkdirs(file_name)
    img = np.array(img)
    cv2.imwrite(file_name, img)
    pass


def process(image, **kwargs):
    save_results = dict(result=[])
    config = kwargs['config']
    # 多尺度测试
    bbox_result = inference_detector(config.model, image)
    win_bboxes = np.empty([0, 6], dtype=np.float32)
    for j in range(len(bbox_result)):
        if len(bbox_result[j]) <= 0 or j == 4:
            continue
        x = np.array([[j] * len(bbox_result[j])])
        bbox_result[j] = np.concatenate([bbox_result[j], x.T], axis=1)
        win_bboxes = np.append(win_bboxes, bbox_result[j], axis=0)
    keep = np.argsort(-win_bboxes[:, 4])[:config.slice_num]
    win_bboxes = win_bboxes[keep]
    # plt_bbox(result['img'], win_bboxes[:, :4], win_bboxes[:, 5], idx)
    if len(win_bboxes) < 1:
        return save_results
    if config.nms_whole:
        mst_bboxes = torch.from_numpy(win_bboxes).float().cuda()
        bboxes = mst_bboxes[:, :4].contiguous()
        scores = mst_bboxes[:, 4].contiguous()
        labels = (mst_bboxes[:, 5].long() + 1).contiguous()
        bboxes, keep = batched_nms(
            bboxes, scores, labels, nms_cfg=config.model.cfg.test_cfg.rcnn.nms)
        labels = labels[keep]
        bboxes = bboxes.cpu().numpy()
    else:
        bboxes = win_bboxes[:, :5]
        labels = win_bboxes[:, 5]
    for r, label in zip(bboxes, labels):
        bbox = list(map(float, r[:4]))
        label = int(label)
        if int(label) not in config.label2name:
            continue
        category_id, score = config.label2name[int(label)], r[4]
        save_results['result'].append({
            'name':
                config.label2name[label],
            'image_id':
                os.path.basename(image)[:-4],
            'xmin':
                bbox[0],
            'ymin':
                bbox[1],
            'xmax':
                bbox[2],
            'ymax':
                bbox[3],
            'confidence':
                float(score)
        })

    end = time.time()
    print('second/img: {:.2f}'.format(end - kwargs['time']['start']))
    kwargs['time']['start'] = end
    return save_results


def parallel_infer():
    config = Config()
    mkdirs(config.save_file)
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(
        tasks=config.tasks,
        process=process,
        process_params=process_params,
        collect=['result'],
        workers_num=8,
        print_process=5)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {} s'.format(end - start))
    results = json_normalize(results['result'])
    results.to_csv(config.save_file, index=False)
    print("process ok!")


if __name__ == '__main__':
    parallel_infer()
