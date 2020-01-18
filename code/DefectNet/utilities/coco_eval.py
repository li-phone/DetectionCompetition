from argparse import ArgumentParser

from mmdet.core import coco_eval

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_evaluate(gt_file, dt_file):
    print("start evaluate using coco api")
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(dt_file)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    # cocoEval.params.iouThrs = np.linspace(0.2, 0.8, 8)
    cocoEval.evaluate()
    cocoEval.accumulate()
    report = cocoEval.summarize()
    return report


def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument(
        '--result',
        default='/home/liphone/undone-work/uselessNet/code/mmdetection/work_dirs/fabric/cascade_rcnn_r50_fpn_1x/baseline/latest_epoch_12_out.pkl.bbox.json',
        help='result file path')
    parser.add_argument(
        '--ann',
        default='/home/liphone/undone-work/data/detection/fabric/annotations/instance_test_34.json',
        help='annotation file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    parser.add_argument(
        '--classwise', default=True, action='store_true', help='whether eval class wise ap')
    args = parser.parse_args()
    coco_eval(args.result, args.types, args.ann, args.max_dets, args.classwise)


if __name__ == '__main__':
    main()
