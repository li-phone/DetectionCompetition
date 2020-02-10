from utilities.utils import save_dict, load_dict


def check_coco(src, dst):
    annotations = load_dict(src)

    categories = [None] * len(annotations['categories'])
    for c in annotations['categories']:
        categories[c['id']] = c
    annotations['categories'] = categories
    for i, r in enumerate(annotations['annotations']):
        r['id'] = i
    save_dict(dst, annotations)


check_coco(
    '/home/liphone/undone-work/data/detection/alcohol/annotations/instances_train_20191223_annotations_bk.json',
    '/home/liphone/undone-work/data/detection/alcohol/annotations/instances_train_20191223_annotations.json'
)
