import glob
import os
import json
import uuid
import pandas as pd
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

try:
    from pandas import json_normalize
except:
    from pandas.io.json import json_normalize


def mkdirs(path, is_file=True):
    if is_file:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def json2node(key, obj):
    e = ET.Element(key)
    if not isinstance(obj, dict):
        e.text = str(obj)
        return e
    for k, v in obj.items():
        if isinstance(v, (list, tuple)):
            for x in v:
                e.append(json2node(k, x))
        else:
            e.append(json2node(k, v))
    return e


def write_xml(o, save_name):
    def __indent(elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                __indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    # root = ET.Element('annotation')  # 创建节点
    root = json2node('annotation', o)
    tree = ET.ElementTree(root)  # 创建文档
    __indent(root)  # 增加换行符
    tree.write(save_name, encoding='utf-8', xml_declaration=True)


def panda2labelme(ann_file, save_dir, maxDets=500, thr=0., category_id=None, **kwargs):
    from fname2id import fname2id, cat2id
    id2fname = {v: k for k, v in fname2id.items()}
    # id2cat = {v: k for k, v in cat2id.items()}
    id2cat = {1: "visible body", 2: "full body", 3: "head", 4: "car"}
    # df = pd.read_json(ann_file)
    with open(ann_file) as fp:
        o = json.load(fp)
    df = json_normalize(o)
    df = df[df['category_id'] == category_id]
    df = df[df['score'] > thr]
    for image_id in tqdm(list(np.unique(df['image_id']))):
        img_df = df[df['image_id'] == image_id]
        keep_df = pd.DataFrame()
        for catid in list(np.unique(img_df['category_id'])):
            cat_df = img_df[img_df['category_id'] == catid]
            cat_df = cat_df.sort_values(by='score', ascending=False, inplace=False)
            cat_df = cat_df[:maxDets]
            keep_df = pd.concat([keep_df, cat_df])
        filename = os.path.basename(id2fname[image_id])
        obj = dict(
            folder=".",
            filename=filename,
            path=filename,
            source=dict(database="Unknown"),
            size=dict(width=0, height=0, depth=3, ),
            segmented=0,
            object=[],
        )
        for i in range(len(keep_df)):
            row = keep_df.iloc[i]
            obj['object'].append(dict(
                name=id2cat[row['category_id']],
                pose="Unspecified",
                truncated=0,
                difficult=0,
                score=row['score'],
                bndbox=dict(
                    xmin=int(row['bbox_left']),
                    ymin=int(row['bbox_top']),
                    xmax=int(row['bbox_left'] + row['bbox_width']),
                    ymax=int(row['bbox_top'] + row['bbox_height']),
                ),
            ))
        save_name = os.path.join(save_dir, os.path.basename(filename))[:-3] + "xml"
        mkdirs(save_name)
        write_xml(obj, save_name)


def parse_args():
    parser = argparse.ArgumentParser(description='Transform other dataset format into coco format')
    parser.add_argument('--x',
                        default=r"work_dirs/track/best-r50-mst_slice-mst_slice-scale_3-score_thr-4.json",
                        help='x file/folder or original annotation file in test_img mode')
    parser.add_argument('--save_name',
                        default=r"panda_round1_test_202104_A",
                        help='save coco filename')
    parser.add_argument(
        '--fmt',
        choices=['json', 'xml', 'test_dir', 'csv', 'PANDA'],
        default='json', help='format type')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.fmt == 'json':
        panda2labelme(args.x, args.save_name, category_id=4)


if __name__ == '__main__':
    main()
