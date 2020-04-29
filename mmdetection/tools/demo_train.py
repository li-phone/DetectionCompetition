from batch_train import BatchTrain


def train_models(test_status=10):
    BatchTrain(cfg_path='../configs/cartoon_face/retinanet_r50_fpn_1x.py',
               data_mode='val', train_sleep_time=0, test_sleep_time=test_status).common_train()
import torchvision.models.resnet

def main():
    train_models()


if __name__ == '__main__':
    main()
