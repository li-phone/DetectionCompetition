from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import json
import argparse
import os
import glob
from tqdm import tqdm
from utilities.draw_util import draw_coco


def save_json(results, submit_filename):
    with open(submit_filename, 'w') as fp:
        json.dump(results, fp, indent=4, separators=(',', ': '))


def infer(model, image_paths, have_bg=False):
    results = dict(images=[], annotations=[])
    # name2label = {1: 1, 9: 2, 5: 3, 3: 4, 4: 5, 0: 6, 2: 7, 8: 8, 6: 9, 10: 10, 7: 11}
    # label2name = {v: k for k, v in name2label.items()}
    for img_id, path in tqdm(enumerate(image_paths)):
        results['images'].append(dict(file_name=os.path.basename(path), id=img_id))
        result = inference_detector(model, path)
        for idx, pred in enumerate(result):
            # category_id = label2name[idx+1]
            if have_bg:
                category_id = idx
            else:
                category_id = idx + 1

            if 0 == category_id:
                continue

            for x in pred:
                bbox_pred = {
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])],
                    "score": float(x[4]),
                }
                results['annotations'].append(bbox_pred)
        # break
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--config',
        default='../config_alcohol/cascade_rcnn_r50_fpn_1x/dig_augment_n4_id3.py',
        help='train config file path')
    parser.add_argument(
        '--resume_from',
        default='../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/dig_augment_n4_id3/epoch_12.pth',
        help='train config file path')
    parser.add_argument(
        '--img_dir',
        default='/home/liphone/undone-work/data/detection/alcohol/test')
    parser.add_argument(
        '--work_dir',
        default='../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/dig_augment_n4_id3/',
        help='train config file path')
    parser.add_argument('--have_bg', default=False)
    args = parser.parse_args()

    return args


def draw(img_dir, work_dir, ann_file):
    defect_name2label = {
        0: '背景',
        1: '瓶盖破损',
        2: '瓶盖变形',
        3: '瓶盖坏边',
        4: '瓶盖打旋',
        5: '瓶盖断点',
        6: '标贴歪斜',
        7: '标贴起皱',
        8: '标贴气泡',
        9: '喷码正常',
        10: '喷码异常'
    }
    label_list = [v for k, v in defect_name2label.items()]
    draw_coco(
        ann_file,
        img_dir,
        os.path.join(work_dir, '.infer_tmp'),
        label_list,
    )


def main(**kwargs):
    args = parse_args()
    for k, v in kwargs.items():
        args.__setattr__(k, v)

    model = init_detector(args.config, args.resume_from, device='cuda:0')
    image_paths = glob.glob(os.path.join(args.img_dir, '*'))

    results = infer(model, image_paths, args.have_bg)
    save_json(results, args.submit_out)
    # draw(args.img_dir, args.work_dir, submit_filename)


if __name__ == '__main__':
    main()
    print('infer all test images ok!')
