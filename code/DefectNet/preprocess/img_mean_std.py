# # coding: utf-8
#
# import numpy as np
# import cv2
# import random
# from tqdm import tqdm
# import glob
#
# train_txt_path = '../../Data/train.txt'
#
# means, stdevs = [], []
# img_paths = glob.glob('' + '/*')
# for img_path in tqdm(img_paths):
#     img = cv2.imread(img_path)
#     # img = cv2.resize(img, (img_h, img_w))
#
#     img = img[:, :, :, np.newaxis]
#     imgs = np.concatenate((imgs, img), axis=3)
#     print(i)
#
# imgs = imgs.astype(np.float32) / 255.
#
# for i in range(3):
#     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))
#
# means.reverse()  # BGR --> RGB
# stdevs.reverse()
#
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))
# print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
