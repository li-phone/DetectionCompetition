from pycocotools.coco import COCO
from torchvision.models import resnet

ann_file = '/home/liphone/undone-work/data/detection/alcohol/annotations/instances_train_20191223_annotations.json'
coco = COCO(ann_file)
img_ids = coco.getImgIds()
for img_id in img_ids:
    ann_ids = coco.getAnnIds(img_id)
    anns = coco.loadAnns(ann_ids)
    bg_cnt = 0
    for ann in anns:
        if ann['category_id'] == 0:
            bg_cnt += 1
    if bg_cnt != 0:
        if bg_cnt > 1:
            print('bg_cng', bg_cnt)
        if bg_cnt != len(anns):
            print('bg_cng/len(anns):', bg_cnt, len(anns))
