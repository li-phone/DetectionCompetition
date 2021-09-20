import glob
from tqdm import tqdm
from orange2_parallel_inference import main as infer_main


def main():
    # configs = glob.glob("../configs/goods/*.py")
    # configs = ["../configs/goods/cas_x101-MST_0.8_0.2-DCNv2-SoftNMS-Translate-ColorTransform.py"]
    configs = [
        # "../configs/orange2/cas_r101-best_base-resize_0.5_1.0-iou_0.5_score_0.01.py",
        # "../configs/orange2/cas_r101-best_base-resize_0.5_1.0-iou_0.5_score_0.001.py",
        # "../configs/orange2/cas_r101-best_base-resize_0.5_1.0-iou_0.5_score_0.3.py",
        "../configs/orange2/cas_r101-best_base-resize_0.5_1.0-iou_0.8_score_0.3.py",
    ]
    configs.sort()
    for cfg in tqdm(configs):
        try:
            infer_main(cfg)
        except Exception as e:
            print(cfg, 'except:', e)
    print('infer_main all config ok!')


if __name__ == '__main__':
    main()
