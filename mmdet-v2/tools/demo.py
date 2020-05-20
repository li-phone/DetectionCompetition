from batch_process import BatchTrain

BatchTrain(
    '../configs/breast/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_quantile.py',
    data_mode='test', train_time=0, test_time=-1
).common_train()
