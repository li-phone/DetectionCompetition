Experiment Records
------------------

#### Cascade R-CNN: 


|序号| 数据集 | 尺寸 | 模型 | tricks |  score | local |
|:----| :-----| :---- | :---- | :---- |:---- | :---- |
|1| train_a-1461 | (960, 720) | cas_r50 | ~~MST(0.5-1.5)~~  |  0.869196  | 0.915 | 
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


         