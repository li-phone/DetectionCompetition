Experiment Records
------------------

#### Cascade R-CNN: 

cas_r50+DCNV2+MST+ATT3+SoftNMS

|序号 | 模型 | tricks |  score | local |
|:----|:---- | :---- |:---- | :---- |
|1| cas_r50 | MST(0.8-1.2) | 0.74501 |  |
|2| cas_r50 | MST(0.8-1.2) - **softNMS** | 0.74726 |  |
|3| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 | 0.74579 |  |
|4| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + **finetune_1(iou_0.9_score_0.8)** | 0.77315 |  |
|5| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + finetune_1(iou_0.9_score_0.7) | 0.76554 |  |
|6| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + **finetune_1(iou_0.8_score_0.8)+SoftNMS** | 0.77607 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + **finetune_1(iou_0.8_score_0.8)-SoftNMS** | 0.78023 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + **finetune_1(iou_0.7_score_0.8)-SoftNMS** | 0.78286 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + **finetune_1(iou_0.6_score_0.8)** | 0.76522 |  |
|7| cas_r101 | MST(0.8-1.2) - softNMS - ATT3 + **finetune_1(iou_0.7_score_0.8)** | 0.61929 |  |
|7| detr-cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + **finetune_1(iou_0.7_score_0.8)** |  |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + finetune_1(iou_0.8_score_0.8)-SoftNMS+aug | 0.75288 |  |
|8| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + finetune_2 | 0.76969 |  |
|9| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + finetune_1 + softNMS | 0.7665 |  |
|10| cas_r50 | MST(0.8-1.2) - softNMS - ATT3 + finetune_1 - DCNV2+SoftNMS | 0.76919 |  |
|11| cas_x101 | MST(0.8-1.2) - softNMS - ATT3 + finetune_1+SoftNMS | 0.72731 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + lr:0.02/8 | 0.77087 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + pafpn | 0.77542 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + roi_14 | 0.76306 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 |  0.78375 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + pseudo_label | 0.77046 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + scale_4 | 0.773 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001 | 0.78402 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001+ratio_7 | 0.77705 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001+fp32 | 0.77249 |  |
|7| detectoRS | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001 | 0.65166 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001+-predict-iou_0.7_score_0.8_iter_1 | 0.77711 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001+-predict-iou_0.8_score_0.8_iter_1 | 0.76664 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001+-predict-iou_0.9_score_0.8_iter_1 | 0.7631 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001 + ATT5| 0.78085 | 0.78085 |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001 + max_per_img_500| 0.78402 |  |
|7| cas_r50 | MST(0.8-1.2) - softNMS + finetune_1(iou_0.7_score_0.8) + iou_0.45 + score_0.0001+giou |  |  |
|2| cas_r101 | MST(0.8-1.2) - **softNMS** |  |  |


|序号| 数据集 | 尺寸 | 模型 | tricks |  score | local |
|:----| :-----| :---- | :---- | :---- |:---- | :---- |
|2| train_a-1461 | (1333, 800) | cas_r50 | MST(0.5-1.5)| 0.872508 | 0.926 |
|3| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.8-1.2)** | 0.881005 | 0.936 |
|4| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.5-1.5)+DCNv2**  | 0.881057 | 0.933 |
|5| train_a-1461 | (1333, 800) | cas_r50 | ~~MST(0.5-1.5)+flipv~~  | 0.866532 | 0.923 |
|6| train_a-1461 | (1333, 800) |cas_r50 | ~~MST(0.5-1.5)+diagonal~~  | 0.866769 | 0.923 |
|7| train_a-1461 | (1333, 800) | cas_r50 | MST(0.5-1.5)+PhotoMetricDistortion  |  |  |
|8| train_a-1461 | (1333, 800) | cas_r50 | ~~MST(0.5-1.5)+Shear~~  | Errno 99 |  |
|9| train_a-1461 | (1333, 800) | cas_r50 | ~~MST(0.5-1.5)+Rotate~~  | 0.845165 | 0.888 |
|10| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.5-1.5)+Translate**  | 0.877453 | 0.924 |
|11| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.5-1.5)+ColorTransform**  | 0.874534 | 0.925 |
|12| train_a-1461 | (1333, 800) | cas_r50 | ~~MST(0.5-1.5)+EqualizeTransform~~  | 0.866053 | 0.921 |
|13| train_a-1461 | (1333, 800) | cas_r50 | ~~MST(0.5-1.5)+BrightnessTransform~~  | 0.834906 | 0.908 |
|14| train_a-1461 | (1333, 800) | cas_r50 | ~~MST(0.5-1.5)+ContrastTransform~~  | 0.841905 | 0.907 |
|15| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.8-1.2)+TTA(MST)** | 0.87775 |  |
|15| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.8-1.2)+Soft-NMS** | 0.881796 |  |
|15| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.8-1.2)+DCNv2<br>+TTA(MST)+Soft-NMS**  | 0.886217 |  |
|15| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.8-1.2)+DCNv2<br>+Soft-NMS**  | 0.887587 |  |
|15| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.8-1.2)+DCNv2<br>+TTA(MST)**  | 0.886806 |  |
|16| train_a-1461 | (1333, 800) | cas_r50 | **MST(0.8-1.2)+DCNv2<br>+Soft-NMS+Translate+ColorTransform**  |  |  |


         