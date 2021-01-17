from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np
import matplotlib.pyplot as plt
import cv2

config_file = '../configs/_raw_/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '../work_dirs/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
imgs = []
for i in range(3):
    img = cv2.imread('demo.jpg')
    imgs.append(img)
res = inference_detector(model, imgs[0])
model.show_result('demo.jpg', res)
plt.show()
print(res)
