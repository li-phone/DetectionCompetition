import glob
from tqdm import tqdm
from orange2_parallel_inference import main as infer_main


def main():
    # configs = glob.glob("../configs/goods/*.py")
    # configs = ["../configs/goods/cas_x101-MST_0.8_0.2-DCNv2-SoftNMS-Translate-ColorTransform.py"]
    # configs = ["../configs/orange/cas_r50-best.py"]
    # configs = ["../configs/orange/cas_r50-best-finetune_data.py"]
    # configs = ["../configs/orange/cas_r50-best-finetune_data-DCNV2.py"]
    configs = [
        # "../configs/orange/cas_r50-best-finetune_data-x101.py",
        # "../configs/orange/cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-finetune_data_iou_0.8_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-finetune_data_iou_0.9_score_0.7_iter_1.py",
        # "../configs/orange/cas_r50-best-finetune_data_iou_0.6_score_0.8_iter_1.py",
        # "../configs/orange/cas_r101v1d-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/detectoRS-cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-lr_0.0125.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-pafpn.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-roi_14.py",
        # "../configs/orange/cas_r101-best.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-pseudo_label.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-scale_4.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-ratio_7.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-giou.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-fp32.py",
        # "../configs/orange/detectoRS-cas_r50-best-finetune_data_iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-iou_0.7_score_0.8-predict-iou_0.7_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-iou_0.7_score_0.8-predict-iou_0.8_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-iou_0.7_score_0.8-predict-iou_0.9_score_0.8_iter_1.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-albu.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-mosaic.py",
        # "../configs/orange/cas_r50-best-iou_0.7_score_0.8-rot90.py",
        # "../configs/orange/doublehead-rpn-3-DIouLoss.py",
        # "../configs/orange2/cas_r50-best-slice_800x800.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000_3000x3000.py",
        # "../configs/orange2/cas_r50-best-slice_1333x800.py",
        # "../configs/orange2/cas_r50-best-slice_1333x1333.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000-img_scale_1666x1000.py",
        # "../configs/orange2/cas_r50-best-slice_1333x1333-img_scale_2221x1333.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000-finetune_iou_0.7_score_0.8.py",
        # "../configs/orange2/cas_r50-best-slice_1000x1000-finetune_iou_0.9_score_0.9.py",
        # "../configs/orange2/cas_r101-best-slice_1000x1000-softnms-aug-800x800.py",
        # "../configs/orange2/cas_x101-best-slice_1000x1000-softnms-aug-800x800.py",
        # "../configs/orange2/cas_r101-best_base-ovlap_0.5.py",
        # "../configs/orange2/cas_r101-best_base-1000x1000_ovlap_0.5.py",
        # "../configs/orange2/cas_r101-best_base-anchor_ratio.py",
        # "../configs/orange2/cas_r101-best_base-ovlap_0.5-auto.py",
        # "../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0_1.5.py",
        "../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0-v2.py",
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
