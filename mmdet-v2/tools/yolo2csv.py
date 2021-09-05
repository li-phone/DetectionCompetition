import os
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm

root = "data/orange2/"
txts = glob.glob(root + "train/labels/*.txt")
img_dir = root + "train/images/"
id2cat = {
    '0': 'bug', '1': 'fruit_bug',
}
not_exist_imgs = []
anns = []
for txt in tqdm(txts):
    img_id = os.path.basename(txt)[:-4]
    img_path = img_dir + img_id + '.jpg'
    if not os.path.exists(img_path):
        img_path = img_dir + img_id + '.JPG'
        if not os.path.exists(img_path):
            not_exist_imgs.append(img_path)
            continue
    img = Image.open(img_path)
    img_h, img_w = img.height, img.width
    with open(txt) as fp:
        lines = fp.readlines()
        for line in lines:
            d = line.strip().split(' ')
            b = list(map(float, d[1:]))
            cx, cy, w, h = b[0] * img_w, b[1] * img_h, b[2] * img_w, b[3] * img_h
            anns.append({
                'file_name': os.path.basename(img_path),
                'label': id2cat[d[0]],
                'xmin': cx - w / 2,
                'ymin': cy - h / 2,
                'xmax': cx + w / 2,
                'ymax': cy + h / 2,
            })
print(not_exist_imgs)
print("not exist number", len(not_exist_imgs))
anns = pd.json_normalize(anns)
if not os.path.exists(root + 'annotations'):
    os.mkdir(root + 'annotations')
anns.to_csv(root + 'annotations/annotations.csv', index=False)
