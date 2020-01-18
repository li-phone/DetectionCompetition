import os
import json
import random
from pycocotools.coco import COCO
import copy


def split_coco(ann_path, save_dir, mode='34', rate=0.6, random_state=666):
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()
    random.seed(random_state)
    random.shuffle(image_ids)

    train_size = int(len(image_ids) * rate)
    train_img_ids = image_ids[:train_size]
    train_image_info = coco.loadImgs(train_img_ids)
    instance_train = copy.deepcopy(coco.dataset)
    instance_train['images'] = train_image_info
    train_img_ids = set(train_img_ids)
    instance_train['annotations'] = [ann for ann in instance_train['annotations'] if ann['image_id'] in train_img_ids]
    save_name = os.path.join(save_dir, 'instance_train_{}.json'.format(mode))
    with open(save_name, 'w') as fp:
        json.dump(instance_train, fp, indent=1, separators=(',', ': '))

    test_img_ids = image_ids[train_size:]
    test_image_info = coco.loadImgs(test_img_ids)
    instance_test = copy.deepcopy(coco.dataset)
    instance_test['images'] = test_image_info
    test_img_ids = set(test_img_ids)
    instance_test['annotations'] = [ann for ann in instance_test['annotations'] if ann['image_id'] in test_img_ids]
    save_name = os.path.join(save_dir, 'instance_test_{}.json'.format(mode))
    with open(save_name, 'w') as fp:
        json.dump(instance_test, fp, indent=1, separators=(',', ': '))


def main():
    ann_path = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_20_all_no_bg.json'
    save_dir = '/home/liphone/undone-work/data/detection/fabric/annotations'
    split_coco(ann_path, save_dir, mode='20')


if __name__ == '__main__':
    main()
