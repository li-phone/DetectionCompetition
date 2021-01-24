from pycocotools.coco import COCO
from pandas import json_normalize
import matplotlib.pyplot as plt
import numpy as np

coco = COCO("/home/lifeng/undone-work/dataset/detection/tile/annotations/instance_all-check.json")
annotations = json_normalize(coco.dataset['annotations'])
rst = annotations.groupby('category_id').count()
rst.plot()
plt.show()

bbox = np.array([b for b in list(annotations['bbox'])])
import pandas as pd

bbox = pd.DataFrame(bbox, columns=['x', 'y', 'w', 'h'])
bbox.plot().scatter(x='w', y='h')
plt.show()

bbox['w'].plot.hist(bins=40, log=True, stacked=True, range=(0, 2000))
plt.show()
bbox['w'].plot.box()
plt.show()

bbox['h'].plot.hist(bins=40, log=True, stacked=True)
plt.show()
bbox['h'].plot.box()
plt.show()

print('ok!')
