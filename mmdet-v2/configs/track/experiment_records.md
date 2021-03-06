Experiment Records
------------------

#### Cascade R-CNN+ResNet: **bs_r50_20e_track.py**

    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.2
    online score: 0.3119
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.298
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.476
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.321
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.102
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.456
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.374
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.374
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.137
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.548
    +--------------+-------+-----------+-------+----------+-------+
    | category     | AP    | category  | AP    | category | AP    |
    +--------------+-------+-----------+-------+----------+-------+
    | car          | 0.392 | full body | 0.320 | head     | 0.212 |
    | visible body | 0.268 | None      | None  | None     | None  |
    +--------------+-------+-----------+-------+----------+-------+
    
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.15
    score: 0.3173

    post_process: nms = dict(type='soft_nms', iou_threshold=0.5), score_thr = 0.15
    score: 0.3173
    
    test: dict(type='Resize', img_scale=(4000, 4000), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 0.2237
    
    test: dict(type='Resize', img_scale=(800, 800), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 0.3267
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.334
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.513
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.354
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.072
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.518
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.387
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.387
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.091
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.590
    +--------------+-------+-----------+-------+----------+-------+
    | category     | AP    | category  | AP    | category | AP    |
    +--------------+-------+-----------+-------+----------+-------+
    | car          | 0.482 | full body | 0.350 | head     | 0.165 |
    | visible body | 0.337 | None      | None  | None     | None  |
    +--------------+-------+-----------+-------+----------+-------+

#### Cascade R-CNN+ResNet: **bs_r101_v1d_20e_track.py**

    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 
        
 
#### Cascade R-CNN+HResNest: **bs_resnest_101_20e_track.py**

    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=[(1333, 640), (1333, 800)], multiscale_mode='range', keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(1333, 800), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 
        

#### Cascade R-CNN+HRNet: **bs_hr18_20e_track.py**

    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 
        