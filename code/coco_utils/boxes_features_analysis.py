import numpy as np
import json
import os
from tqdm import tqdm
import glob
import pandas as pd
import cv2 as cv
import time
import visdom
import matplotlib.pyplot as plt
import seaborn as sns


def draw_box_features(anno_list_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    annos_df = pd.read_json(anno_list_path)
    print(annos_df.describe())

    # 画散点图
    sns.set()
    sns.relplot(x="w", y="h", col="label_name", data=annos_df)
    plt.savefig(save_dir + "/scatter.svg")
    plt.show()

    sns.lmplot(x="w", y="h", col="label_name", data=annos_df)
    plt.savefig(save_dir + "/lmplot.svg")
    plt.show()

    sns.pairplot(data=annos_df, hue="label_name")
    plt.savefig(save_dir + "/pairplot.png")
    plt.show()

    # 面积直方图
    areas = np.array(annos_df['area'])
    sns.distplot(areas, axlabel="area")
    plt.savefig(save_dir + "/area.svg")
    plt.show()

    # # 纵横比直方图
    aspect_ratios = np.array(annos_df['h']) / np.array(annos_df['w'])
    aspect_ratios = aspect_ratios[aspect_ratios != np.inf]
    sns.distplot(aspect_ratios, axlabel="aspect_ratios")
    plt.savefig(save_dir + "/aspect_ratios.svg")
    plt.show()

    # # 宽长比直方图
    width_height_ratios = np.array(annos_df['w']) / np.array(annos_df['h'])
    width_height_ratios = width_height_ratios[width_height_ratios != np.inf]
    sns.distplot(width_height_ratios, axlabel="width_height_ratios")
    plt.savefig(save_dir + "/width_height_ratios.svg")
    plt.show()

    # 面积比直方图
    scales = np.array(areas) / np.mean(areas)
    scales = scales[scales != np.inf]
    sns.distplot(scales, axlabel="scales")
    plt.savefig(save_dir + "/scales.svg")
    plt.show()


if __name__ == "__main__":
    draw_box_features(
        "../coco/coco_coco/annotations/coco_list_train.json",
        "../coco/tools/imgs",
    )
    pass
