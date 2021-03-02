Experiment Records
------------------
#### Submit Results

    baseline: **bs_r50_20e_track.py**
    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.2
    score: 0.3119
    
    
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.15
    score: 0.3173


    post_process: nms = dict(type='soft_nms', iou_threshold=0.5), score_thr = 0.15
    score: 0.3173
    
    
    test: dict(type='Resize', img_scale=(4000, 4000), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.15
    score: 
        