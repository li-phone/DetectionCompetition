from pycocotools.coco import COCO
import pandas as pd
import os
from pandas import json_normalize
from utilities.utils import load_dict


def coco2csvsubmit(gt_result, dt_result, csv_name, file_type='.xml'):
    if isinstance(gt_result, str):
        gt_result = COCO(gt_result)
    if isinstance(dt_result, str):
        dt_result = load_dict(dt_result)
    cat2label = {r['id']: r['name'] for r in gt_result.dataset['categories']}
    imgid2filename = {r['id']: r['file_name'] for r in dt_result['images']}
    results = []
    for ann in dt_result['annotations']:
        x, y, w, h = tuple(ann['bbox'])
        xmin, ymin, xmax, ymax = int(x + 0.5), int(y + 0.5), int(x + w + 0.5), int(y + h + 0.5)
        image_id = imgid2filename[ann['image_id']]
        image_id = os.path.basename(image_id)
        image_id = image_id[:image_id.rfind('.')] + file_type
        row = dict(
            name=cat2label[ann['category_id']],
            image_id=image_id,
            confidence=ann['score'],
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
        )
        results.append(row)
    results = json_normalize(results)
    results.to_csv(csv_name, index=False)


def coco_defect2csv(ann_path, save_name):
    coco = COCO(ann_path)
    dataset = coco.dataset

    true_nums = have_defect(dataset, dataset['images'])
    y_true = [0 if x == 0 else 1 for x in true_nums]
    ids = [r['file_name'] for r in dataset['images']]
    data = pd.DataFrame({'id': ids, 'label': y_true})
    data.to_csv(save_name, index=False)


def coco2csv(ann_path, save_name):
    coco = COCO(ann_path)
    filenames, labels = [], []
    cat2label = {v['id']: v['name'] for v in coco.dataset['categories']}
    for i, image in enumerate(coco.dataset['images']):
        img_id = image['id']
        ann_id = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_id)
        for v in anns:
            if cat2label[v['category_id']] != '桔子皮':
                filenames.append(os.path.basename(image['file_name']))
                labels.append(v['category_id'])
    train_csv = pd.DataFrame(
        data={
            'file_name': filenames,
            'label': labels
        }
    )
    train_csv.to_csv(save_name, index=False)


if __name__ == '__main__':
    # coco_defect2csv(
    #     '/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.json',
    #     '/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.csv'
    # )
    coco2csv(
        '/home/liphone/undone-work/defectNet/DefectNet/work_dirs/garbage_data/instance_train.json',
        '/home/liphone/undone-work/defectNet/DefectNet/work_dirs/garbage_data/train-debug.csv'
    )
