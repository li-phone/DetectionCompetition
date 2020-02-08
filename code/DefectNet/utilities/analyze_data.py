import seaborn as sns
import matplotlib.pyplot as plt
import json
from pandas.io.json import json_normalize
import os
import pandas as pd
import numpy as np


def read_json(json_path):
    results = []
    np_names = [
        'AP', 'AP:0.50', 'AP:0.75', 'AP:S', 'AP:M', 'AP:L',
        'AR', 'AR:0.50', 'AR:0.75', 'AR:S', 'AR:M', 'AR:L',
    ]
    with open(json_path) as fp:
        lines = fp.readlines()
        for line in lines:
            r = json.loads(line)
            d = r['data']['bbox']['data']
            result = dict(cfg=r['cfg'], mode=r['mode'])
            for k, v in zip(np_names, d['coco_eval']):
                result[k] = v
            for k, v in d['classwise'].items():
                result[k] = v
            for k1, v1 in d['defect_eval'].items():
                if isinstance(v1, list):
                    result[k1] = np.mean(v1)
                elif isinstance(v1, dict):
                    for k2, v2 in v1['macro avg'].items():
                        result[k2] = v2
            results.append(result)
    return results


def phrase_json(json_path):
    save_path = json_path[:-5] + '.csv'
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    results = read_json(json_path)
    df = json_normalize(results)
    df.to_csv(save_path, index=False)
    return df


def get_sns_data(data, x_name, y_names, type):
    x, y, hue = np.empty(0), np.empty(0), np.empty(0)
    for y_name in y_names:
        x = np.append(x, data[x_name])
        y = np.append(y, data[y_name])
        hue = np.append(hue, [type[y_name]] * data[y_name].shape[0])
    return pd.DataFrame(dict(x=x, y=y, type=hue))


def draw_figure():
    save_dir = '../results/imgs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = phrase_json('../config_alcohol/cascade_rcnn_r50_fpn_1x/eval_alcohol_dataset_report.json')
    sns.set(style="darkgrid")
    ids = []
    for i in range(data.shape[0]):
        r = data.iloc[i]
        arrs = r['cfg'].split('_')
        ids.append(float(arrs[1][:-1]))

    x_name = 'defect finding weight'
    data[x_name] = ids
    data = data[data['mode'] == 'test']

    fig = plt.figure(figsize=(6.4 * 3, 4.8))
    axs = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    def draw_ap_weight(ax):
        y_names = ['AP', 'AP:0.50']
        type = {'AP': 'IoU=0.50:0.95', 'AP:0.50': 'IoU=0.50'}
        sns_data = get_sns_data(data, x_name, y_names, type)
        new_x, new_y = 'defect finding weight', 'average precision'
        sns_data = sns_data.rename(columns={'x': new_x, 'y': new_y})
        ax = sns.lineplot(
            ax=ax,
            x=new_x, y=new_y,
            hue="type",
            style="type",
            markers=True,
            dashes=False,
            data=sns_data,
            ci=None
        )
        # plt.savefig('../results/imgs/AP-defect_finding_weight.svg')
        # plt.show()

    draw_ap_weight(axs[0])

    def draw_f1_score_weight(ax):
        y_names = ['f1-score']
        type = {'f1-score': 'f1-score'}
        sns_data = get_sns_data(data, x_name, y_names, type)
        new_x, new_y = 'defect finding weight', 'macro average f1-score'
        sns_data = sns_data.rename(columns={'x': new_x, 'y': new_y})
        ax = sns.lineplot(
            ax=ax,
            x=new_x, y=new_y,
            hue="type",
            style="type",
            markers=True,
            dashes=False,
            data=sns_data,
            ci=None
        )
        # plt.savefig('../results/imgs/f1_score-defect_finding_weight.svg')
        # plt.show()

    draw_f1_score_weight(axs[1])

    def draw_speed_weight(ax):
        y_names = ['fps', 'defect_fps', 'normal_fps']
        type = dict(fps='all images', defect_fps='defect images', normal_fps='normal images')
        sns_data = get_sns_data(data, x_name, y_names, type)
        new_x, new_y = 'defect finding weight', 'average time(ms)'
        sns_data = sns_data.rename(columns={'x': new_x, 'y': new_y})
        ax = sns.lineplot(
            ax=ax,
            x=new_x, y=new_y,
            hue="type",
            style="type",
            markers=True,
            dashes=False,
            data=sns_data,
            ci=None
        )
        # plt.savefig('../results/imgs/speed-defect_finding_weight.svg')
        # plt.show()

    draw_speed_weight(axs[2])
    plt.savefig(save_dir + '/result-defect_finding_weight.eps')
    plt.show()


def count_data(ann_file, head=None):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    defect_nums = np.empty(0, dtype=int)
    for image in coco.dataset['images']:
        cnt = 0
        annIds = coco.getAnnIds(imgIds=image['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if ann['category_id'] != 0:
                cnt += 1
        defect_nums = np.append(defect_nums, cnt)
    normal_cnt = np.where(defect_nums == 0)[0]
    if head is not None:
        print(head + ':\n')
    print('All images count:', len(coco.dataset['images']))
    print('Normal images count:', normal_cnt.shape[0])
    print('Defect images count:', defect_nums.shape[0] - normal_cnt.shape[0])


def main():
    count_data('/home/liphone/undone-work/data/detection/alcohol/annotations/instances_train_20191223_annotations.json',
               'all')
    count_data('/home/liphone/undone-work/data/detection/alcohol/annotations/instance_train_alcohol.json', 'train')
    count_data('/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.json', 'test')
    draw_figure()


if __name__ == '__main__':
    main()
