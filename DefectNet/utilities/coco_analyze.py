import seaborn as sns
import matplotlib.pyplot as plt
import json
from pandas.io.json import json_normalize
import os
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def count_image(coco, ignore_id=0):
    defect_nums = np.empty(0, dtype=int)
    for image in coco.dataset['images']:
        cnt = 0
        annIds = coco.getAnnIds(imgIds=image['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if ann['category_id'] != ignore_id:
                cnt += 1
        defect_nums = np.append(defect_nums, cnt)
    normal_shape = np.where(defect_nums == 0)[0]
    all_cnt, normal_cnt = len(coco.dataset['images']), normal_shape.shape[0]
    defect_cnt = defect_nums.shape[0] - normal_shape.shape[0]
    assert normal_cnt + defect_cnt == all_cnt
    return all_cnt, normal_cnt, defect_cnt


def chg2coco(coco):
    if isinstance(coco, str):
        coco = COCO(coco)
    return coco


def save_plt(save_name, file_types=None):
    if file_types is None:
        file_types = ['.svg', '.jpg', '.eps']
    for t in file_types:
        plt.savefig(save_name[:-4] + t)


class COCOAnalysis(object):
    # matplotlib.style.use('ggplot')  # 使用ggplot样式 %matplotlib inline
    sns.set(style="darkgrid")

    save_img_dir = '../results/fabric_defect_detection/'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    def image_count_summary(coco):
        coco = chg2coco(coco)
        img_cnts = count_image(coco)
        print('total images: {}, normal images: {}, defect images: {}, normal : defective: {}'
              .format(img_cnts[0], img_cnts[1], img_cnts[2], img_cnts[1] / img_cnts[2]))
        print('total defect number: {}\n'
              .format(len(coco.dataset['annotations'])))

    def category_distribution(coco, legends=None, cn2eng=None):
        if isinstance(coco, str):
            cocos = [coco]
        else:
            cocos = coco
        cat_dists = pd.DataFrame()
        for i, coco in enumerate(cocos):
            coco = chg2coco(coco)
            dataset = coco.dataset
            if cn2eng is not None:
                cat2label = {r['id']: cn2eng[r['name']] for r in dataset['categories']}
            else:
                cat2label = {r['id']: r['name'] for r in dataset['categories']}
            for ann in dataset['annotations']:
                ann['category_id'] = cat2label[ann['category_id']]

            ann_df = json_normalize(dataset['annotations'])
            cat_dist = ann_df['category_id'].value_counts()
            cat_dist = cat_dist.drop('background')
            if legends is not None:
                cat_dist = pd.DataFrame(data={legends[i]: cat_dist})
            else:
                cat_dist = pd.DataFrame(data=cat_dist)
            cat_dists = pd.concat([cat_dists, cat_dist], axis=1, sort=True)
        cat_dists = cat_dists.sort_values(by=legends[0], ascending=True)
        pplt = cat_dists.plot.barh(stacked=True)
        plt.xlabel('number of defect categories')
        plt.subplots_adjust(left=0.27, right=0.97, top=0.96)
        save_plt(save_img_dir + 'category_distribution.jpg')
        plt.show()

    def bbox_distribution(coco):
        coco = chg2coco(coco)
        dataset = coco.dataset
        boxes = [b['bbox'] for b in dataset['annotations']]
        box_df = pd.DataFrame(data=boxes, columns=['x', 'y', 'bbox width', 'bbox height'])
        box_df.plot(kind="scatter", x="bbox width", y="bbox height", alpha=0.2)
        from tricks.data_cluster import box_cluster
        boxes_ = box_cluster(dataset, n=10)
        save_plt(save_img_dir + 'bbox_distribution.jpg')
        plt.show()


def main():
    ann_files = [
        '/home/liphone/undone-work/data/detection/fabric/annotations/instance_train,type=34,.json',
        '/home/liphone/undone-work/data/detection/fabric/annotations/instance_train_rate=0.80.json',
        '/home/liphone/undone-work/data/detection/fabric/annotations/instance_test_rate=0.80.json',
    ]
    image_count_summary(ann_files[0])
    image_count_summary(ann_files[1])
    image_count_summary(ann_files[2])

    bbox_distribution(ann_files[1])

    cn2eng = {
        '背景': 'background', '破洞': 'hole', '水渍': 'water stain', '油渍': 'oil stain',
        '污渍': 'soiled', '三丝': 'three silk', '结头': 'knots', '花板跳': 'card skip', '百脚': 'mispick',
        '毛粒': 'card neps', '粗经': 'coarse end', '松经': 'loose warp', '断经': 'cracked ends',
        '吊经': 'buttonhold selvage', '粗维': 'coarse picks', '纬缩': 'looped weft', '浆斑': 'hard size',
        '整经结': 'warping knot', '星跳': 'stitch', '跳花': 'skips',
        '断氨纶': 'broken spandex', '稀密档': 'thin thick place', '浪纹档': 'buckling place', '色差档': 'color shading',
        '磨痕': 'smash', '轧痕': 'roll marks', '修痕': 'take marks', '烧毛痕': 'singeing', '死皱': 'crinked',
        '云织': 'uneven weaving', '双纬': 'double pick', '双经': 'double end', '跳纱': 'felter', '筘路': 'reediness',
        '纬纱不良': 'bad weft yarn',
    }
    category_distribution(ann_files, cn2eng=cn2eng, legends=['all', 'train', 'test'])


if __name__ == '__main__':
    main()
