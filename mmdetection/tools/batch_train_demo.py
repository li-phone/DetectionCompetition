from batch_train import BatchTrain


def train_models(test_status=-10):
    # train for one-model method without background
    BatchTrain(cfg_path='../configs/bottle/one_model_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()


def main():
    train_models()
    pass


if __name__ == '__main__':
    main()
