from argparse import ArgumentParser

from mmdet.core import coco_eval


def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument('--result',
                        default='/home/liphone/undone-work/DetCompetition/mmdet-v1/work_dirs/garbage_huawei/cascade_rcnn_x101_64x4d_fpn_1x+multiscale+softnms+flip/data_mode=test+.bbox.json',
                        help='result file path')
    parser.add_argument('--ann', default='/home/liphone/undone-work/data/detection/garbage_huawei/annotations/instance_train.json',
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
