import os
import time
import json
import cv2
import glob
import torch
import numpy as np
from mmcv.ops.nms import batched_nms
from mmdet.datasets.pipelines import Compose
from mmdet.third_party.parallel import Parallel
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


class Config(object):
    # process module
    # 1000 x 1000
    train_pipeline1 = [
        dict(type='LoadImageFromFile'),
        dict(
            type='SliceImage',
            base_win=(1000, 1000),
            step=(0.5, 0.5),
            resize=(1, 1),
            keep_none=True),
    ]
    # train_pipeline2 = [
    #     dict(type='LoadImageFromFile'),
    #     dict(
    #         type='SliceImage',
    #         base_win=(1000, 1000),
    #         step=(0.5, 0.5),
    #         resize=(1, 1),
    #         keep_none=True),
    # ]
    # 2000 x 2000
    # train_pipeline2 = [
    #     dict(type='LoadImageFromFile'),
    #     dict(
    #         type='SliceImage',
    #         base_win=(2000, 2000),
    #         step=(0.2, 0.2),
    #         resize=(1, 1),
    #         keep_none=True),
    # ]
    composes = [
        # Compose(train_pipeline2),
        Compose(train_pipeline1)
    ]
    show_result_on_image = True
    nms_whole = True
    max_slice_num = np.array([15000 * 10, 7500 * 10])[::-1]

    # data module
    img_dir = "data/orange2/test_A/images"
    # label2name = {x['id']: x['name'] for x in original_coco.dataset['categories']}
    label2name = {0: 0, 1: 1}
    device = 'cuda:0'

    def __init__(self, cfg=None):
        self.tasks = glob.glob(self.img_dir + "/*")
        self.config_file = cfg
        self.checkpoint_file = 'work_dirs/' + os.path.basename(cfg)[:-3] + '/latest.pth'
        self.save_file = f"__work_dirs__/orange2/{os.path.basename(cfg)[:-3]}/detection-results"
        self.model = init_detector(cfg, self.checkpoint_file, device=self.device)


def mkdirs(path, is_file=True):
    if is_file:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def plt_bbox(image, boxes, labels, prefix=None, threshold=0.5):
    # show test image
    import matplotlib.pyplot as plt
    from mmcv.visualization.image import imshow_det_bboxes
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    img = np.array(img)
    index = np.where(boxes[:, 4] >= threshold)
    boxes = boxes[index]
    labels = labels[index]
    img = imshow_det_bboxes(img, boxes[:, :4], labels, show=False)
    # plt.imshow(img)
    if isinstance(image, str):
        file_name = f"__show_img__/{image}.jpg"
    else:
        file_name = f"__show_img__/{prefix}.jpg"
    mkdirs(file_name)
    img = np.array(img)
    cv2.imwrite(file_name, img)
    pass


def process(image, **kwargs):
    save_results = dict(result=[])
    config = kwargs['config']

    # 多尺度测试
    mst_bboxes = np.empty([0, 6], dtype=np.float32)
    idx = 0
    for compose, slice_num in zip(config.composes, config.max_slice_num):
        results = compose({'img_prefix': config.img_dir, 'img_info': {'filename': os.path.basename(image)}})
        if results is None or len(results) == 0: continue
        if not isinstance(results, (list, tuple)): results = [results]
        img_bboxes = np.empty([0, 6], dtype=np.float32)
        for i, result in enumerate(results):
            bbox_result = inference_detector(config.model, result['img'])

            if config.show_result_on_image:
                show_labels = []
                show_bboxes = np.empty([0, 5], dtype=np.float32)
                for j2 in range(len(bbox_result)):
                    show_bboxes = np.append(show_bboxes, bbox_result[j2], axis=0)
                    show_labels.extend([j2] * len(bbox_result[j2]))
                show_labels = np.array(show_labels)
                plt_bbox(result['img'], show_bboxes, show_labels, prefix=f"{image}-" + "{:03d}".format(i))

            win_bboxes = np.empty([0, 6], dtype=np.float32)
            for j in range(len(bbox_result)):
                if len(bbox_result[j]) <= 0:
                    continue
                if 'is_slice' in result and result['is_slice']:
                    w, h = compose.transforms[1].base_win
                    keep_idx = []
                    for _, b in enumerate(bbox_result[j]):
                        if 0 <= b[0] and b[2] <= w and 0 <= b[1] and b[3] <= h:
                            keep_idx.append(_)
                    bbox_result[j] = bbox_result[j][keep_idx]
                if 'slice_roi__left_top' in result:
                    bbox_result[j][:, 0] += result['slice_roi__left_top'][0]
                    bbox_result[j][:, 1] += result['slice_roi__left_top'][1]
                    bbox_result[j][:, 2] += result['slice_roi__left_top'][0]
                    bbox_result[j][:, 3] += result['slice_roi__left_top'][1]
                if 'slice_image__window' in result:
                    bbox_result[j][:, 0] += result['slice_image__window'][0]
                    bbox_result[j][:, 1] += result['slice_image__window'][1]
                    bbox_result[j][:, 2] += result['slice_image__window'][0]
                    bbox_result[j][:, 3] += result['slice_image__window'][1]
                x = np.array([[j] * len(bbox_result[j])])
                bbox_result[j] = np.concatenate([bbox_result[j], x.T], axis=1)
                win_bboxes = np.append(win_bboxes, bbox_result[j], axis=0)
            keep = np.argsort(-win_bboxes[:, 4])[:slice_num]
            win_bboxes = win_bboxes[keep]
            idx += 1
            img_bboxes = np.append(img_bboxes, win_bboxes, axis=0)
        img_bboxes[:, 0] = img_bboxes[:, 0] / compose.transforms[1].resize[0]
        img_bboxes[:, 1] = img_bboxes[:, 1] / compose.transforms[1].resize[1]
        img_bboxes[:, 2] = img_bboxes[:, 2] / compose.transforms[1].resize[0]
        img_bboxes[:, 3] = img_bboxes[:, 3] / compose.transforms[1].resize[1]
        mst_bboxes = np.append(mst_bboxes, img_bboxes, axis=0)
    if len(mst_bboxes) < 1:
        return save_results

    if config.nms_whole:
        mst_bboxes = torch.from_numpy(mst_bboxes).float().cuda()
        bboxes = mst_bboxes[:, :4].contiguous()
        scores = mst_bboxes[:, 4].contiguous()
        labels = (mst_bboxes[:, 5].long()).contiguous()
        bboxes, keep = batched_nms(
            bboxes, scores, labels, nms_cfg=config.model.cfg.test_cfg.rcnn.nms)
        labels = labels[keep]
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        bboxes = mst_bboxes[:, :5]
        labels = mst_bboxes[:, 5]

    if config.show_result_on_image:
        plt_bbox(image, bboxes, labels, prefix=f"{image}")

    per_result = []
    for r, label in zip(bboxes, labels):
        bbox, score = list(map(float, r[:4])), r[4]
        per_result.append([config.label2name[int(label)], score, bbox[0], bbox[1], bbox[2], bbox[3]])
    key = os.path.basename(image)
    save_results['result'].append({key: per_result})

    end = time.time()
    print('second/img: {:.2f}'.format(end - kwargs['time']['start']))
    kwargs['time']['start'] = end
    return save_results


def parallel_infer(cfg=None):
    config = Config(cfg)
    mkdirs(config.save_file, is_file=False)
    process_params = dict(config=config, time=dict(start=time.time()))
    settings = dict(
        tasks=config.tasks,
        process=process,
        process_params=process_params,
        collect=['result'],
        workers_num=12,
        print_process=10)
    parallel = Parallel(**settings)
    start = time.time()
    results = parallel()
    end = time.time()
    print('times: {} s'.format(end - start))
    for r1 in results['result']:
        for k, v in r1.items():
            file_name = os.path.join(config.save_file, k[:-4] + '.txt')
            with open(file_name, "w") as fp:
                lines = []
                for r2 in v:
                    line = str(r2).replace("'", '')
                    line = line[1:-1].replace(', ', ' ') + '\n'
                    lines.append(line)
                fp.writelines(lines)
    pwd_dir = os.getcwd()
    ch_dir = config.save_file[:config.save_file.rfind('/')]
    os.chdir(ch_dir)
    os.system(f"zip detection-results.zip -r detection-results")
    os.chdir(pwd_dir)
    print("process ok!")


def main(config=None):
    parallel_infer(config)


if __name__ == '__main__':
    main()
