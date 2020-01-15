# batch train
from tqdm import tqdm
import os
import time

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)

from coco_eval import coco_evaluate


def hint(wav_file='./wav/qq.wav', n=3):
    import pygame
    for i in range(n):
        pygame.mixer.init()
        pygame.mixer.music.load(wav_file)
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()


def main():
    train_params = [
        # 'focal_loss',
        # 'anchor_cluster',
        # '4_stage',
        'iou_decrease',
        'iou_increase',
        'iou_cluster',
        'baseline',
    ]
    for p in tqdm(train_params):
        print('p={}'.format(p))

        # train
        cmd = 'python train.py --config ../config_alcohol/cascade_rcnn_r50_fpn_1x/{}.py'.format(p)
        ret = os.system(cmd)
        print('{} train successfully!'.format(p))
        hint()

        # test
        line = 'python test.py --eval bbox ' \
               + ' --config             ../config_alcohol/cascade_rcnn_r50_fpn_1x/{}.py' \
               + ' --checkpoint         ../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/{}/latest.pth' \
               + ' --out                ../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_out.pkl' \
               + ' --json_out           ../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_json_out.json'
        cmd = line.format(p, p, p, p)
        try:
            ret = os.system(cmd)
            print('{} test successfully!'.format(p))
        except:
            print('{} test successfully!'.format(p))
        hint()

        # coco eval
        report = coco_evaluate(
            '/home/liphone/undone-work/data/detection/alcohol/annotations/instances_train_20191223_annotations.json',
            '../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_out.pkl.bbox.json'.format(p)
        )
        with open('../config_alcohol/cascade_rcnn_r50_fpn_1x/eval_report.txt', 'a+') as fp:
            line = '\n\n' + '=' * 36 + p + '=' * 36
            fp.write(line + report)
        hint()

        # infer
        line = 'python infer.py --config ../config_alcohol/cascade_rcnn_r50_fpn_1x/{}.py' \
               + ' --resume_from         ../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/{}/epoch_12.pth' \
               + ' --img_dir             /home/liphone/undone-work/data/detection/alcohol/test' \
               + ' --work_dir            ../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/{}/'
        cmd = line.format(p, p, p, p)
        ret = os.system(cmd)
        hint()

        time.sleep(1800)


if __name__ == '__main__':
    main()
