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
    data_name = 'alcohol'
    train_modes = [
        # 'baseline_have_bg',
        # 'baseline_no_bg',
        # 'baseline_20_have_bg',
        # 'baseline_20_no_bg',
        # 'baseline_34_have_bg',
        # 'baseline_34_no_bg',
        # 'DefectNet_have_bg',
        'DefectNet_no_bg_4_weight',
    ]
    train_params = [
        # 'instance_train_alcohol.json',
        # 'instance_train_alcohol_nobg.json',
        # 'instance_train_fabric_20.json',
        # 'instance_train_fabric_20_nobg.json',
        # 'instance_train_fabric_34.json',
        # 'instance_train_fabric_34_nobg.json',
        'none',
    ]
    for p, ann in tqdm(zip(train_modes, train_params)):
        print('p={}'.format(p))

        # train
        cmd = 'python train.py --config ../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/{}.py'.format(p)
        ret = os.system(cmd)
        print('{} train successfully!'.format(p))
        hint()

        # # test for train set
        # line = 'python test.py --eval bbox ' \
        #        + ' --config     ../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/{}.py' \
        #        + ' --checkpoint ../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/epoch_12.pth' \
        #        + ' --out        ../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_out.pkl' \
        #        + ' --json_out   ../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_json_out.json'
        # cmd = line.format(p, p, p, p)
        # try:
        #     # ret = os.system(cmd)
        #     print('{} test successfully!'.format(p))
        # except:
        #     print('{} test successfully!'.format(p))
        # hint()

        # # coco eval for train set
        # ann_file = '/home/liphone/undone-work/data/detection/' + data_name + '/annotations/{}'.format(ann)
        # result_file = '../work_dirs/' + data_name \
        #               + '/cascade_rcnn_r50_fpn_1x/{}/latest_epoch_12_out.pkl.bbox.json'.format(p)
        # reports = coco_eval(result_file, ['bbox'], ann_file, classwise=True)
        # with open('../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/eval_train_report.txt', 'a+') as fp:
        #     line = '\n\n' + '=' * 36 + p + '=' * 36 + '\n'
        #     fp.write(line)
        #     for rpt in reports:
        #         fp.write(rpt[0] + '\n' + rpt[1] + '\n')
        # hint()

        # infer for test set
        from infer import infer_main
        kargs = dict(
            config='../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/{}.py'.format(p),
            resume_from='../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}/latest.pth'.format(p),
            ann_file='/home/liphone/undone-work/data/detection/' + data_name + '/annotations/instance_test_alcohol.json',
            img_dir='/home/liphone/undone-work/data/detection/' + data_name + '/trainval'.format(p),
            work_dir='../work_dirs/' + data_name + '/cascade_rcnn_r50_fpn_1x/{}'.format(p),
        )
        rpts = infer_main(**kargs)
        with open('../config_' + data_name + '/cascade_rcnn_r50_fpn_1x/eval_test_report.txt', 'a+') as fp:
            line = '\n\n' + '=' * 36 + p + '=' * 36 + '\n'
            fp.write(line)
            for k, v in rpts.items():
                fp.write(str(k) + ':\n' + str(v) + '\n')
        hint()

        time.sleep(1800)


if __name__ == '__main__':
    main()
