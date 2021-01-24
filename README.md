
## Introduction

This repository inherits from MMDetection which is an open source object detection toolbox based on PyTorch 
and also is a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).


### Highlights

#### third_party

- **Parallel**

    我们增加了使用多线程去完成多任务资源的处理接口

#### datasets/pipelines

- **Slice Image**

    我们增加了Slice Image管道，包括两个处理过程：
    * SliceROI：使用canny算法从图像中切割出感兴趣的区域
    * SliceImage：使用滑动窗口方法从图像中切割成若干张小图片

- **Compose**

    我们修改了compose使得它能够支持多个data的处理

#### tools/3rd_party

- **x2coco**

    我们增加了多种格式转换成coco格式的工具，包括Pascal VOC（xml）、天池（json）、测试图片目录转coco伪标签等
    
- **COCO Check**

    我们增加了检查coco原始标注是否正确的工具，同时提供基于opencv imshow接口的修正功能
    
- **COCO Split**

    我们增加了将coco数据集划分成训练数据和验证数据的工具
    
- **Parallel Inference**

    我们增加了多线程切图推理工具，同时也包括了后处理的NMS操作

- **Do Submit**

    我们增加了coco测试输出结果转换成若干种格式提交的工具，包括天池（单个json文件）、和鲸（多个json文件）等格式

- **Parallel Slice**

    我们增加了多线程将图像切割成若干张小图片的工具

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog
None

## Contributions
[li-phone](https://github.com/li-phone)
