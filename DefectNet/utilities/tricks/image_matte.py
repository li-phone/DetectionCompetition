import numpy as np
import os
import cv2 as cv
import copy


def _matte(img_src, box_src, img_dst, box_dst, mixup_ratio=0.5, img_mixup=False, shift=True):
    img_src = img_src.astype(np.float) / 255.0
    img_dst = img_dst.astype(np.float) / 255.0
    box_dst = np.array(box_dst)
    new_h, new_w = max(img_src.shape[0], img_dst.shape[0]), max(img_src.shape[1], img_dst.shape[1])
    new_img = np.zeros((new_h, new_w, 3))
    img_dst_h, img_dst_w, img_dst_c = img_dst.shape
    img_src_h, img_src_w, img_src_c = img_src.shape
    if img_mixup:
        new_img[:img_src_h, :img_src_w, :] = mixup_ratio * img_src + 0.0 * new_img[:img_src_h, :img_src_w, :]
        new_img[:img_dst_h, :img_dst_w, :] = (1.0 - mixup_ratio) * img_dst + 1.0 * new_img[:img_dst_h, :img_dst_w, :]
    else:
        new_img[:img_dst_h, :img_dst_w, :] = img_dst
    for i, sbox in enumerate(box_src):
        bi = [int(x + 0.5) for x in sbox]
        wmin, hmin, wmax, hmax = bi
        simg_h, simg_w, simg_c = img_src.shape
        wmin, hmin, wmax, hmax = max(0, wmin), max(0, hmin), min(wmax, simg_w), min(hmax, simg_h)
        bbox_img = img_src[hmin:hmax, wmin:wmax, :].copy()
        if mixup_ratio > 0:
            if shift:
                box_w, box_h = (wmax - wmin), (hmax - hmin)
                hmin = np.random.randint(0, new_h - box_h)
                wmin = np.random.randint(0, new_w - box_w)
                hmax, wmax = hmin + box_h, wmin + box_w
            new_img[hmin:hmax, wmin:wmax, :] = mixup_ratio * bbox_img + (1.0 - mixup_ratio) * new_img[hmin:hmax,
                                                                                              wmin:wmax, :]
        else:
            box_w, box_h = (wmax - wmin), (hmax - hmin)
            left, top, right, bottom = np.min(box_dst[:, 0]), np.min(box_dst[:, 1]), np.max(box_dst[:, 2]), np.max(
                box_dst[:, 3])
            free = [left, top, new_w - right, new_h - bottom]
            ind = int(np.argmax(free))
            ratio_ = free[ind] / max(box_w, box_h)
            ratio_ = min(1, ratio_)
            size_ = (int(box_w * ratio_ * np.random.random()), int(box_h * ratio_ * np.random.random()))
            bbox_img = cv.resize(bbox_img, size_)
            _h, _w, _c = bbox_img.shape
            if ind == 0:
                wmin = left - _w
            elif ind == 1:
                hmin = top - _h
            elif ind == 2:
                wmin = right
            elif ind == 3:
                hmin = bottom
            hmin, wmin = int(hmin), int(wmin)
            hmax, wmax = hmin + _h, wmin + _w
            new_img[hmin:hmax, wmin:wmax, :] = bbox_img
        new_box = np.array([wmin, hmin, wmax, hmax]).reshape((-1, 4))
        box_dst = np.append(box_dst, new_box, axis=0)
    new_img = np.array(new_img * 255).astype(np.uint8)
    return new_img, box_dst


def main():
    def random_choose(id, n):
        img = cv.imread('/home/liphone/undone-work/data/detection/garbage/train/images/00ab57885fc4.png')
        box = [[100, 100, 200, 200]]
        return img, box, [1]

    from pycocotools.coco import COCO
    from tqdm import tqdm
    img_dir = '/home/liphone/undone-work/data/detection/garbage/train/images'
    new_img_dir = '/home/liphone/undone-work/data/detection/garbage/train/new_images'
    n = 5
    coco = COCO('/home/liphone/undone-work/data/detection/garbage/train/instance_train.json')
    dataset = coco.dataset
    new_anns = []
    for image in dataset['images']:
        img_id = image['id']
        ann_id = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_id)
        box_dst = [a['bbox'] for a in anns]
        box_dst = np.array(box_dst)
        box_dst[:, 3] += box_dst[:, 1]
        box_dst[:, 2] += box_dst[:, 0]
        img_dst_path = os.path.join(img_dir, image['file_name'])
        img_dst = cv.imread(img_dst_path)
        img_src, box_src, label = random_choose(0, n)
        new_img, new_box = _matte(img_src, box_src, img_dst, box_dst, mixup_ratio=-1)
        cv.imwrite(os.path.join(new_img_dir, image['file_name']), new_img)
        for i in range(box_dst.shape[0], new_box.shape[0]):
            ann_ = copy.deepcopy(anns[0])
            b = new_box[i]
            ann_['area'] = [(b[3] - b[1]) * b[2] - b[0]]
            idx = i - box_dst.shape[0]
            ann_['category_id'] = label[idx]
            ann_['id'] = (len(dataset['annotations']) + 1)
            dataset['annotations'].append(ann_)
    dataset['annotations'] = new_anns
    with open('/home/liphone/undone-work/data/detection/garbage/train/new_instance_train.json', 'w') as fp:
        import json
        json.dump(dataset, fp)


if __name__ == '__main__':
    main()
