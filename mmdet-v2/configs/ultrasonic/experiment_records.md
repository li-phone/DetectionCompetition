Experiment Records
------------------

#### Cascade R-CNN: 

**best score: 0.52947355**

    baseline: best-r50-mst_slice.py    
    
    + ResNet50
    + DCNv2
    + model fp16
    + IoU fp16
    + nms=dict(type='soft_nms', iou_thr=0.5)
    + img_scale=(1333, 800), ratio_range=(0.8, 1.2)
    + img_scale=[(1066, 800), (1333, 800), (1600, 800)], flip=True
    + lr=0.02 / 4
    + step=[16, 19], epoch=20
    
    score: 0.52947355    
    
    --------------------------------------------------------------------------------
    best-x101-mst_slice.py
    
    + resnext101_64x4d
    + ...
    
    score:     
    
      
            