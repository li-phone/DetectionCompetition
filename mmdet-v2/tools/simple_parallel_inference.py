import os
import time
import cv2
import torch
import glob
import json
import numpy as np
from pandas import json_normalize
from pandas.io.json import json_normalize
from mmcv.ops.nms import batched_nms
from mmdet.third_party.parallel import Parallel
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


class Config(object):
    # data module
    img_dir = "data/goods/A/test/a_images/"
    slice_num = 10000
    nms_whole = False
    label2name = {0: 'echinus', 1: 'holothurian', 2: 'scallop', 3: 'starfish', 4: 'waterweeds'}

    # inference module
    device = 'cuda:0'

    def __init__(self, cfg=None):
        with open("data/goods/A/test/a_annotations.json") as fp: test_data = json.load(fp)
        self.tasks = test_data['images']
        self.config_file = cfg
        self.checkpoint_file = 'work_dirs/' + os.path.basename(cfg)[:-3] + '/latest.pth'
        self.save_file = "work_dirs/goods/" + os.path.basename(cfg)[:-3] + '.json'
        self.model = init_detector(cfg, self.checkpoint_file, device=self.device)


def mkdirs(path, is_file=True):
    if is_file:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def plt_bbox(image, boxes, labels, idx=None):
    # show test image
    import matplotlib.pyplot as plt
    from mmcv.visualization.image import imshow_det_bboxes
    if isinstance(image, str):
        img = cv2.imread(image)
    img = np.array(img)
    img = imshow_det_bboxes(img, boxes, labels, show=False)
    # plt.imshow(img)
    file_name = "__show_img__/" + '{}'.format(image) + '.jpg'
    mkdirs(file_name)
    img = np.array(img)
    cv2.imwrite(file_name, img)
    pass


def process(image, **kwargs):
    save_results = dict(result=[])
    config = kwargs['config']
    img_file = config.img_dir + image['file_name']
    bbox_result = inference_detector(config.model, img_file)
    win_bboxes = np.empty([0, 6], dtype=np.float32)
    for j in range(len(bbox_result)):
        if len(bbox_result[j]) <= 0:
            continue
        x = np.array([[j] * len(bbox_result[j])])
        bbox_result[j] = np.concatenate([bbox_result[j], x.T], axis=1)
        win_bboxes = np.append(win_bboxes, bbox_result[j], axis=0)
    keep = np.argsort(-win_bboxes[:, 4])[:config.slice_num]
    win_bboxes = win_bboxes[keep]
    plt_bbox(img_file, win_bboxes[:, :4], win_bboxes[:, 5])
    if len(win_bboxes) < 1:
        return save_results
    if config.nms_whole:
        mst_bboxes = torch.from_numpy(win_bboxes).float().cuda()
        bboxes = mst_bboxes[:, :4].contiguous()
        scores = mst_bboxes[:, 4].contiguous()
        labels = (mst_bboxes[:, 5].long()).contiguous()
        bboxes, keep = batched_nms(
            bboxes, scores, labels, nms_cfg=config.model.cfg.test_cfg.rcnn.nms)
        labels = labels[keep]
        bboxes = bboxes.cpu().numpy()
    else:
        bboxes = win_bboxes[:, :5]
        labels = win_bboxes[:, 5]
    for r, label in zip(bboxes, labels):
        bbox = list(map(float, r[:4]))
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        label = int(label)
        # if int(label) not in config.label2name:
        #     continue
        category_id, score = None, r[4]
        save_results['result'].append({
            'image_id': image['id'],
            'category_id': label,
            'bbox': bbox,
            'score': float(score)
        })
    end = time.time()
    # print('second/img: {:.2f}'.format(end - kwargs['time']['start']))
    kwargs['time']['start'] = end
    return save_results


def parallel_infer(cfg=None):
    config = Config(cfg)
    mkdirs(config.save_file)
    images = [{"file_name": x["file_name"], "id": x["id"]} for x in config.tasks]
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(
        tasks=config.tasks,
        process=process,
        process_params=process_params,
        collect=['result'],
        workers_num=6,
        print_process=100)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {} s'.format(end - start))
    results = {"images": images, "annotations": results['result']}
    with open(config.save_file, "w") as fp: json.dump(results, fp)
    print("process ok!")


def main(config=None):
    parallel_infer(config)


if __name__ == '__main__':
    main()
