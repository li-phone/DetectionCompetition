# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


def hint(wav_file='./wav/qq.wav', n=3):
    import pygame
    for i in range(n):
        pygame.mixer.init()
        pygame.mixer.music.load(wav_file)
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()


def main():
    data_name = 'fabric'
    train_params = [
        'baseline_34_no_bg',
        # 'baseline_34_bg',
        # 'DefectNet_34',
        'baseline_20_no_bg',
        # 'baseline_20_bg',
        # 'DefectNet_20',
    ]
    for p, type in tqdm(train_params, [34, 20]):
        print('p={}'.format(p))

        # train
        cmd = 'python train.py --config ../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/{}.py'.format(p)
        ret = os.system(cmd)
        print('{} train successfully!'.format(p))
        hint()

        # test for train set
        line = 'python test.py --eval bbox ' \
               + ' --config     ../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/{}.py' \
               + ' --checkpoint ../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/latest.pth' \
               + ' --out        ../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_out.pkl' \
               + ' --json_out   ../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_json_out.json'
        cmd = line.format(p, p, p, p)
        try:
            ret = os.system(cmd)
            print('{} test successfully!'.format(p))
        except:
            print('{} test successfully!'.format(p))
        hint()

        # coco eval for train set
        ann_file = '/home/liphone/undone-work/data/detection/' + data_name + '/annotations/instance_train_{}.json'.format(
            type)
        result_file = '../work_dirs/' + data_name \
                      + '/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_out.pkl.bbox.json'.format(p)
        reports = coco_eval(result_file, ['bbox'], ann_file, classwise=True)
        with open('../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/eval_train_report.txt', 'a+') as fp:
            line = '\n\n' + '=' * 36 + p + '=' * 36
            fp.write(line)
            for rpt in reports:
                fp.write(rpt[0] + '\n' + rpt[1] + '\n')
        hint()

        # infer for test set
        from infer import infer_main
        kargs = dict(
            config='../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/{}.py'.format(p),
            resume_from='../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/epoch_12.pth'.format(p),
            ann_file='/home/liphone/undone-work/data/detection/' + data_name + '/annotations/instance_test_{}.json'.format(
                type),
            img_dir='/home/liphone/undone-work/data/detection/' + data_name + '/test'.format(p),
            work_dir='../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}'.format(p),
        )
        rpts = infer_main(**kargs)
        with open('../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/eval_test_report.txt', 'a+') as fp:
            line = '\n\n' + '=' * 36 + p + '=' * 36
            fp.write(line)
            for k, v in rpts.items():
                fp.write(k + ':\n' + v + '\n')
        hint()

        time.sleep(1800)


if __name__ == '__main__':
    main()
