import json
import glob
import os.path as osp
from mmdet.apis.inference import inference_detector, init_detector


class Config(object):
    device = 'cuda:0'
    config_file = 'configs/patterned-fabric/bs_x101_64x4d_20e.py'
    checkpoint_file = 'work_dirs/patterned-fabric/bs_x101_64x4d_20e/epoch_24.pth'
    model = init_detector(config_file, checkpoint_file, device=device)
    lab2cat = {
        0: 15, 1: 3, 2: 1, 3: 11, 4: 8, 5: 10,
        6: 5, 7: 6, 8: 14, 9: 13, 10: 4,
        11: 7, 12: 12, 13: 9, 14: 2
    }


def no_parallel_inference():
    config = Config()
    results = []
    images = glob.glob("./tcdata/guangdong1_round2_testB_20191024/*OK/*OK.jpg")
    for image in images:
        result = inference_detector(config.model, image)
        for i, bboxes in enumerate(result):
            for row in bboxes:
                label = config.lab2cat[i]
                bbox, score = [float(x) for x in row[:4]], float(row[4])
                results.append({'name': osp.basename(image), 'category': int(label), 'bbox': bbox, 'score': score})
    with open('result.json', 'w') as fp:
        # json.dump(results, fp, indent=4, separators=(',', ': '))
        json.dump(results, fp)
    print("process ok!")


if __name__ == '__main__':
    no_parallel_inference()
