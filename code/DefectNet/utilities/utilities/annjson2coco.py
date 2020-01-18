import os
import json
import numpy as np
from tqdm import tqdm
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


def gen_coco(ann_json, save_ann_name, img_dir, name2category):
    categories = []
    name2label_20 = {
        '破洞': 1, '水渍_油渍_污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
        '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳_跳花': 16,
        '断氨纶': 17, '稀密档_浪纹档_色差档': 18, '磨痕_轧痕_修痕_烧毛痕': 19,
        '死皱_云织_双纬_双经_跳纱_筘路_纬纱不良': 20,
    }
    for k, v in name2label_20.items():
        categories.append(dict(name=k, id=v, supercategory='fabric_defect'))

    images = {}
    annotations = {}
    for ann in tqdm(ann_json):
        file_name = ann['name']
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            continue

        if file_name not in images:
            images[file_name] = dict(file_name=file_name, id=len(images), width=2446, height=1000)

        bbox, name = ann['bbox'], ann['defect_name']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotations[len(annotations)] = dict(
            id=len(annotations),
            image_id=images[file_name]['id'],
            category_id=name2category[name],
            bbox=_get_box(points),
            iscrowd=0,
            area=area
        )
    images = [v for k, v in images.items()]
    annotations = [v for k, v in annotations.items()]
    instance = dict(info='fabric_defect', license='none', images=images, annotations=annotations, categories=categories)
    with open(save_ann_name, 'w') as fp:
        json.dump(instance, fp, indent=1, separators=(',', ': '))


def main():
    ann_file = '/home/liphone/undone-work/data/detection/fabric/annotations/anno_train_20190818-20190928.json'
    save_ann_name = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_20_all_no_bg.json'
    img_dir = '/home/liphone/undone-work/data/detection/fabric/trainval'
    with open(ann_file) as fp:
        ann_json = json.load(fp)
    # normal_images = glob.glob(os.path.join(img_dir, 'normal_Images_*.jpg'))
    # bbox = [0, 0, 5, 5]
    # for p in normal_images:
    #     ann_json.append(dict(name=os.path.basename(p), defect_name='背景', bbox=bbox))
    # defect_name2label = {
    #     '背景': 0, '破洞': 1, '水渍': 2, '油渍': 3, '污渍': 4, '三丝': 5, '结头': 6, '花板跳': 7, '百脚': 8, '毛粒': 9,
    #     '粗经': 10, '松经': 11, '断经': 12, '吊经': 13, '粗维': 14, '纬缩': 15, '浆斑': 16, '整经结': 17, '星跳': 18, '跳花': 19,
    #     '断氨纶': 20, '稀密档': 21, '浪纹档': 22, '色差档': 23, '磨痕': 24, '轧痕': 25, '修痕': 26, '烧毛痕': 27, '死皱': 28, '云织': 29,
    #     '双纬': 30, '双经': 31, '跳纱': 32, '筘路': 33, '纬纱不良': 34,
    # }
    defect_name2label = {
        '背景': 0, '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
        '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
        '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
        '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
    }
    gen_coco(ann_json, save_ann_name, img_dir, defect_name2label)
    from draw_util import draw_coco
    name2label_20 = {
        '背景': 0, '破洞': 1, '水渍_油渍_污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
        '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳_跳花': 16,
        '断氨纶': 17, '稀密档_浪纹档_色差档': 18, '磨痕_轧痕_修痕_烧毛痕': 19,
        '死皱_云织_双纬_双经_跳纱_筘路_纬纱不良': 20,
    }
    label_list = [k for k, v in name2label_20.items()]
    draw_coco(
        save_ann_name, img_dir, '/home/liphone/undone-work/data/detection/fabric/.draw_coco_20_no_bg', label_list
    )


if __name__ == '__main__':
    main()
