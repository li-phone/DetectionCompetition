import glob
import os.path as osp
import shutil
from tqdm import tqdm

ROOT = '/home/lifeng/data/detection/patterned-fabric/guangdong1_round2_train_part3_20190924/'
SAVE_DIR = "/home/lifeng/data/detection/patterned-fabric/images"
NORMAL_DIR = "/home/lifeng/data/detection/patterned-fabric/normal_images"
TEMPLATE_DIR = "/home/lifeng/data/detection/patterned-fabric/templates"

files = glob.glob(ROOT + "defect/*/*.jpg")
for file in tqdm(files):
    if 'template_' in osp.basename(file):
        dst = osp.join(TEMPLATE_DIR, osp.basename(file))
    else:
        dst = osp.join(SAVE_DIR, osp.basename(file))
    shutil.move(file, dst)

files = glob.glob(ROOT + "normal/*/*.jpg")
for file in tqdm(files):
    if 'template_' in osp.basename(file):
        dst = osp.join(TEMPLATE_DIR, osp.basename(file))
    else:
        dst = osp.join(NORMAL_DIR, 'normal_image_' + osp.basename(file))
    shutil.move(file, dst)
print('ok!')
