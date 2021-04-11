import json
import matplotlib.pyplot as plt
import math
import numpy as np

data = json.load(open('./under_water_train_all.json'))
classes = ["holothurian", "echinus", "scallop", "starfish", "waterweeds"]
# 统计每个类别的目标数目
small = [0 for i in range(len(classes))]
medium = [0 for i in range(len(classes))]
large = [0 for i in range(len(classes))]
# 统计每张图像目标个数
object_per_image = dict()
# 统计目标面积
area = []
for annotation in data['annotations']:
    if object_per_image.get(annotation['image_id']):
        object_per_image[annotation['image_id']] += 1
    else:
        object_per_image[annotation['image_id']] = 1
    box = annotation['bbox']
    area.append(math.sqrt(box[2]*box[3]))
    if box[2] * box[3] < 32 * 32:
        small[annotation['category_id']] += 1
    elif 32 * 32 < box[2] * box[3] < 96 * 96:
        medium[annotation['category_id']] += 1
    else:
        large[annotation['category_id']] += 1
# 目标分布
plt.bar(range(len(classes)), small, color='#66c2a5', tick_label=classes, label='small')
plt.bar(range(len(classes)), medium, color='#8da0cb', bottom=small, label='medium')
plt.bar(range(len(classes)), medium, color='orange', bottom=medium, label='large')
plt.xlabel('class')
plt.ylabel('number')
plt.legend()
plt.grid(axis='y')
plt.show()

# 图像目标分布
plt.hist(object_per_image.values(), bins=30, label='number of eah image')
plt.grid(axis='x')
plt.xlabel('object number')
plt.ylabel('image number')
plt.show()

# 目标面积分布
area = sorted(area)
mean = np.mean(np.array(area))
std = np.mean(np.array(area))
min_area = area[0]
max_area = area[-1]
plt.plot(range(len(area)), area, label=f'min:{round(min_area,2)}\n'
                                       f'max:{round(max_area, 2)}\n'
                                       f'mean:{round(mean, 2)}\n'
                                       f'std:{round(std, 2)}')
plt.grid(axis='y')
plt.legend()
plt.show()

# 面积箱线图
plt.boxplot(area)
plt.grid(axis='y')
plt.show()
