from train_with_tricks import BatchTrain


def main():
    batrian = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/fabric.py', data_mode='test')
    batrian.baseline_train()
    batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)], ratio_range=[0.5, 1.5])
    batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)], multiscale_mode='value')
    batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)])
    batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)])
    batrian.multi_scale_train(img_scale=[(2446, 1000)])


if __name__ == '__main__':
    main()
