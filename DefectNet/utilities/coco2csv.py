from defect_test import have_defect
from pycocotools.coco import COCO
import pandas as pd


def main(ann_path, save_name):
    coco = COCO(ann_path)
    dataset = coco.dataset

    true_nums = have_defect(dataset, dataset['images'])
    y_true = [0 if x == 0 else 1 for x in true_nums]
    ids = [r['file_name'] for r in dataset['images']]
    data = pd.DataFrame({'id': ids, 'label': y_true})
    data.to_csv(save_name, index=False)


if __name__ == '__main__':
    main(
        '/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.json',
        '/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.csv'
    )
