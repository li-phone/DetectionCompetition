Experiment Records
------------------

#### Cascade R-CNN+ResNet50: 

**best score: 0.5113**

    baseline: bs_r50_20e_track.py
    
    slice: window=(4000, 4000), step=(3500, 3500)
    infer: window=(4000, 4000), step=(3500, 3500)
    
    train: dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    test: dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.2
    score: 0.3119
    
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
    
    --------------------------------------------------------------------------------
    bs_r50_20e_overlap_sampler_x2_track.py

    + sampler_x2
        lr: 5e-5
    score: 0.3267
    
    + max_per_img=1000 （发现少部分图片存在约900多个目标）
    score: 0.3134
        
    + **overlap=0.7**
    + **clip_border=False**
    score: 0.3336
    
    (+) MST
    test:  img_scale=[(600, 600), (800, 800), (1000, 1000)], flip=True
    score: 0.3569
    
    --------------------------------------------------------------------------------
    bs_r50_20e_overlap_sampler_x2_mst_track.py

    + sampler_x2
    + overlap=0.7    
    + MST
    train: dict(type='Resize', img_scale=(800, 800), ratio_range=(0.8, 1.2), keep_ratio=True)
    test:  img_scale=[(600, 600), (800, 800), (1000, 1000)], flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 0.3826
    
    + Soft-NMS
    test rcnn: nms=dict(type='soft_nms', iou_thr=0.5), score_thr=0.001
    post_process: nms = dict(type='soft_nms', iou_threshold=0.5), score_thr = 0.001
    score: 0.3861
    
    --------------------------------------------------------------------------------
    bs_r50_all_category_overlap_sampler_x2_track.py

    + sampler_x2
    + overlap=0.7    
    + **all categories**
    train: dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    test:  img_scale=(800, 800), flip=True,
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 0.3474    
    
    --------------------------------------------------------------------------------
    bs_r50_all_category_overlap_sampler_x2_mst_track.py

    + sampler_x2
    + overlap=0.7   
    + all categories
    + **MST**
    train: dict(type='Resize', img_scale=(800, 800), ratio_range=(0.8, 1.2), keep_ratio=True),
    test:  img_scale=[(600, 600), (800, 800), (1000, 1000)], flip=True,
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.005
    score: 0.3888
    
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.002
    score: 0.3893
    
    + **Soft-NMS**
    test rcnn: nms=dict(type='soft_nms', iou_thr=0.5), score_thr=0.001
    post_process: nms = dict(type='soft_nms', iou_threshold=0.5), score_thr = 0.005
    score: 0.3943
    
    --------------------------------------------------------------------------------
    bs_r50_all_cat_ovlap_samp_x2_mst_dcn_track.py
    
    + **dcn-v2**
    post_process: nms = dict(type='soft_nms', iou_threshold=0.5), score_thr = 0.002
    score: 0.4000, 0.3560, 0.4564
           0.3943, 0.3493, 0.4527             
           
    --------------------------------------------------------------------------------
    bs_r50_all_cat_ovlap_samp_x2_mst_dcn_track.py
    
    + **all data train**
    score: 0.4048
          
    + **mst slice test**
    1: 4000 x 4000
    2: 8000 x 8000
    score: 0.4106

    + **mst slice test**
    0: 2000 x 2000
    1: 4000 x 4000
    2: 8000 x 8000
    score: 0.4623
    
    --------------------------------------------------------------------------------
    best-r50-mst_slice.py
    
    + **mst slice test**
    0: 2000 x 2000
    1: 4000 x 4000
    2: 8000 x 8000
    max_slice_num = np.array([300, 150, 75])
    score: 0.4944
    
    + **soft-nms**
    score: 0.5113
    
    

#### Cascade R-CNN+ResNet: **bs_r101_v1d_20e_track.py**

    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(800, 800), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.306
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.515
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.303
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.050
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.478
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.368
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.368
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.074
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.559    
    +--------------+-------+-----------+-------+----------+-------+
    | category     | AP    | category  | AP    | category | AP    |
    +--------------+-------+-----------+-------+----------+-------+
    | car          | 0.456 | full body | 0.324 | head     | 0.139 |
    | visible body | 0.307 | None      | None  | None     | None  |
    +--------------+-------+-----------+-------+----------+-------+
    score: 0.2520
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.10
    score: 0.2520
        
 
#### Cascade R-CNN+ResNest: **bs_resnest_101_20e_track.py**

    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=[(800, 800), (800, 800)], multiscale_mode='range', keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(800, 800), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.355
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.567
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.372
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.091
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.536
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.418
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.418
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.124
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.618
    +--------------+-------+-----------+-------+----------+-------+
    | category     | AP    | category  | AP    | category | AP    |
    +--------------+-------+-----------+-------+----------+-------+
    | car          | 0.490 | full body | 0.392 | head     | 0.160 |
    | visible body | 0.380 | None      | None  | None     | None  |
    +--------------+-------+-----------+-------+----------+-------+
    (2 * 0.355 * 0.418) / (0.355 + 0.418) = 0.38393272
    score: 0.2980
    
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.10
    score: 
        

#### Cascade R-CNN+bs_x101_64x4d_20e_track: **bs_x101_64x4d_20e_track.py**

    slice: window=(4000, 4000), step=(3500, 3500)
    train: dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    infer: window=(4000, 4000), step=(3500, 3500)
    test: dict(type='Resize', img_scale=(800, 800), keep_ratio=True), flip=True
    post_process: nms = dict(type='nms', iou_threshold=0.5), score_thr = 0.001
    score: 0.2892
        