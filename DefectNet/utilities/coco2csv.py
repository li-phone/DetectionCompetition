from defect_test import have_defect
from pycocotools.coco import COCO
import pandas as pd
import os


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
