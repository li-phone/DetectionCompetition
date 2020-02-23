import os
import json
import numpy as np
from tqdm import tqdm
from pandas.io.json import json_normalize
import glob


def _get_box(points):
    min_x = min_y = np.inf
    max_x = max_y = 0
    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def transform2coco(anns, save_name, label2cat):
    if isinstance(anns, str):
        with open(anns) as fp:
            anns = json.load(fp)
    if isinstance(anns, list):
        anns = json_normalize(anns)
        if 'name' in list(anns.columns):
            anns = anns.rename(columns={'name': 'file_name'})
        if 'defect_name' in list(anns.columns):
            anns = anns.rename(columns={'defect_name': 'label'})
        anns['id'] = list(range(anns.shape[0]))

    coco = dict(info='fabric detect detection', license='null', categories=[], images=[], annotations=[])
    if isinstance(label2cat, list):
        coco['categories'] = [dict(name=v, id=i + 1, supercategory='fabric_defect') for i, v in enumerate(label2cat)]
    elif isinstance(label2cat, dict):
        coco['categories'] = [dict(name=k, id=i, supercategory='fabric_defect') for k, i in label2cat.items()]

    images = list(anns['file_name'].unique())
    coco['images'] = [dict(file_name=v, id=i, width=2446, height=1000) for i, v in enumerate(images)]
    image2id = {v: i for i, v in enumerate(images)}

    annotations = anns.to_dict('id')
    for k, v in annotations.items():
        bbox = v['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        ann = dict(
            id=v['id'],
            image_id=image2id[v['file_name']],
            category_id=label2cat[v['label']],
            bbox=_get_box(points),
            iscrowd=0,
            area=area
        )
        coco['annotations'].append(ann)
    with open(save_name, 'w') as fp:
        json.dump(coco, fp, indent=1, separators=(',', ': '))


def main():
    ann_file = '/home/liphone/undone-work/data/detection/fabric/annotations/anno_train_20190818-20190928.json'
    save_ann_name = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_train,type=34,.json'
    img_dir = '/home/liphone/undone-work/data/detection/fabric/trainval'
    with open(ann_file) as fp:
        ann_json = json.load(fp)
    normal_images = glob.glob(os.path.join(img_dir, 'normal_Images_*.jpg'))
    bbox = [0, 0, 32, 32]
    for p in normal_images:
        ann_json.append(dict(name=os.path.basename(p), defect_name='背景', bbox=bbox))
    label2cat = {
        '背景': 0, '破洞': 1, '水渍': 2, '油渍': 3, '污渍': 4, '三丝': 5, '结头': 6, '花板跳': 7, '百脚': 8, '毛粒': 9,
        '粗经': 10, '松经': 11, '断经': 12, '吊经': 13, '粗维': 14, '纬缩': 15, '浆斑': 16, '整经结': 17, '星跳': 18, '跳花': 19,
        '断氨纶': 20, '稀密档': 21, '浪纹档': 22, '色差档': 23, '磨痕': 24, '轧痕': 25, '修痕': 26, '烧毛痕': 27, '死皱': 28, '云织': 29,
        '双纬': 30, '双经': 31, '跳纱': 32, '筘路': 33, '纬纱不良': 34,
    }
    # defect_name2label = {
    #     '背景': 0, '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
    #     '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
    #     '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
    #     '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
    # }
    # name2label_20 = {
    #     '背景': 0, '破洞': 1, '水渍_油渍_污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
    #     '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳_跳花': 16,
    #     '断氨纶': 17, '稀密档_浪纹档_色差档': 18, '磨痕_轧痕_修痕_烧毛痕': 19,
    #     '死皱_云织_双纬_双经_跳纱_筘路_纬纱不良': 20,
    # }
    transform2coco(ann_json, save_ann_name, label2cat)
    from draw_util import draw_coco
    draw_coco(
        save_ann_name, img_dir, '/home/liphone/undone-work/data/detection/fabric/.instance_train,type=34,',
    )


if __name__ == '__main__':
    main()
