from mmdet.apis import inference_detector, init_detector

m = init_detector(
    '../configs/breast/cascade_rcnn_x101_64x4d_fpn_1x.py',
    '../work_dirs/breast/cascade_rcnn_x101_64x4d_fpn_1x+multiscale+softnms/epoch_12.pth')
a = inference_detector(m, '../../modelarts_deploy/model/input/0a6122e077b2.png')
print(a)