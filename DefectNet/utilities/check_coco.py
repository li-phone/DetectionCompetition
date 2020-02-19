from utilities.utils import save_dict, load_dict

from pandas.io.json import json_normalize
import os


def check_coco(src, dst):
    if os.path.exists(dst):
        print('There is an existed {}.'.format(dst))
        return
    coco = load_dict(src)
    cats = json_normalize(coco['categories'])
    cats = cats.sort_values(by='id')
    cats = cats.to_dict('id')
    coco['categories'] = list(cats.values())

    imgs = json_normalize(coco['images'])
    if 'image_id' in list(imgs.columns):
        imgs = imgs.rename(columns={'image_id': 'id'})
    imgs['file_name'] = [os.path.basename(p) for p in list(imgs['file_name'])]
    imgs = imgs.sort_values(by='id')
    imgs = imgs.to_dict('id')
    coco['images'] = list(imgs.values())

    anns = json_normalize(coco['annotations'])
    anns['id'] = list(range(anns.shape[0]))
    anns = anns.to_dict('id')
    coco['annotations'] = list(anns.values())

    save_dict(dst, coco)


def draw(img_dir, work_dir, ann_file):
    coco = load_dict(ann_file)
    label_list = [r['name'] for r in coco['categories']]
    label_list.insert(0, '背景')
    from utilities.draw_util import draw_coco
    draw_coco(
        ann_file,
        img_dir,
        os.path.join(work_dir, '.draw_tmp'),
        label_list,
    )


def check_image(img_dir):
    import cv2 as cv
    import glob
    from tqdm import tqdm
    img_paths = glob.glob(img_dir + '/*')
    for p in tqdm(img_paths):
        img = cv.imread(p)
        cv.imwrite(p, img)


check_image('/home/liphone/undone-work/data/detection/garbage/train/images', )
check_coco(
    '/home/liphone/undone-work/data/detection/garbage/train/train.json',
    '/home/liphone/undone-work/data/detection/garbage/train/instance_train.json'
)
# draw(
#     '/home/liphone/undone-work/data/detection/garbage/train/images',
#     '/home/liphone/undone-work/data/detection/garbage/train/',
#     '/home/liphone/undone-work/data/detection/garbage/train/instance_train.json',
# )
