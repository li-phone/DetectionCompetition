import numpy as np
from pandas.io.json import json_normalize


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def C_n_2(s, ind, result):
    if ind == len(s) - 1:
        return result
    for i in range(ind + 1, len(s)):
        result.append([s[ind], s[i]])
    return C_n_2(s, ind + 1, result)


def overlap_iou(anns, min_iou=0.):
    if isinstance(anns, dict):
        anns = json_normalize(anns)
    cats = anns['category'].unique()
    cp_cats = []
    cp_cats = C_n_2(cats, 0, cp_cats)
    for cat in cp_cats:
        ann1 = anns[anns['category'] == cat[0]]
        ann2 = anns[anns['category'] == cat[1]]

    pass


def main():
    if isinstance(anns, dict):
        anns = json_normalize(anns)
    pass


if __name__ == '__main__':
    main()
