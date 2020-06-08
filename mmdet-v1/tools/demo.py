from batch_train import BatchTrain


def train_models(test_status=10):
    BatchTrain(cfg_path='../configs/garbage_huawei/cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()
    # BatchTrain(cfg_path='../configs/breast/cascade_rcnn_x101_64x4d_fpn_1x_anchor_ratios.py',
    #            data_mode='val', train_sleep_time=0, test_sleep_time=test_status).common_train()
    # BatchTrain(cfg_path='../configs/garbage_huawei/cascade_rcnn_cbnet_tb_r50_fpn_1x.py',
    #            data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()


def main():
    train_models()


if __name__ == '__main__':
    main()
