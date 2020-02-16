# -*- coding:utf-8 -*-
import os.path as osp
from torch.utils import data
from torchvision import transforms as T
import torch
import numpy as np
from PIL import Image
import cv2
import importlib
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import shuffle
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def collate_fn(batch):
    if (isinstance(batch[0], tuple) or isinstance(batch[0], list)) and len(batch[0]) != 1:
        return tuple(zip(*batch))
    else:
        return tuple(batch)


class DataReader(data.Dataset):

    def __init__(self, ann_files, img_dirs, transform=None, mode=None, img_scale=(224, 224), keep_ratio=False):
        self.ann_files = ann_files
        self.img_dirs = img_dirs
        self.mode = mode
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

        if transform is None:
            if self.mode == 'train':
                self.transforms = T.Compose([
                    T.Resize(img_scale),
                    # T.CenterCrop((224, 224)),
                    # T.RandomHorizontalFlip(0.5),
                    # T.RandomVerticalFlip(0.5),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(img_scale),
                    # T.CenterCrop((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transform
        self._read_ann_file()

    def _read_ann_file(self):
        self.image_paths = []
        self.annotations = []
        if self.ann_files is None:
            for d in self.img_dirs:
                self.image_paths.extend(glob.glob(os.path.join(d, '*')))
        else:
            for ann_file, img_dir in zip(self.ann_files, self.img_dirs):
                with open(ann_file, 'r') as f:
                    lines = f.readlines()
                    lines = lines[1:]
                    for line in lines:
                        item = line.strip().split(',')
                        self.image_paths.append(os.path.join(img_dir, item[0]))
                        self.annotations.append(item[1])

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert('RGB')
        img = self.transforms(img)
        if len(self.annotations) == 0:
            return img
        else:
            target = self.annotations[index]
            target = int(target)
            return img, target

    def __len__(self):
        return len(self.image_paths)