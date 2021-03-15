from pandas.io.json import json_normalize
from pandas import json_normalize
import json
import numpy as np
from mmcv.ops.nms import soft_nms


def summary(file):
    with open(file, 'r') as fp:
        o = json.load(fp)
    boxes = json_normalize(o['annotations'])
    print('-' * 32, 'image_size=', len(set(boxes['image_id'])), '-' * 32)
    print('-' * 32, 'category_id', '-' * 32)
    print(boxes.groupby(by='category_id').count().sort_values(by='id', ascending=False).head(10))
    print('-' * 32, 'image_id', '-' * 32)
    image_ids = boxes.groupby(by='image_id').count().sort_values(by='id', ascending=False)
    print(image_ids.head(5))

    def func(df, thr):
        print('-' * 64)
        if 0 <= thr <= 1:
            print('percent, ', thr)
            thr = df['id'][int((1 - thr) * len(df))]
        print('thr', str(thr))
        print('> {} size '.format(thr), len(image_ids[image_ids['id'] >= thr]))
        print('< {} size '.format(thr), len(image_ids[image_ids['id'] < thr]))

    func(image_ids, 512 * 0.25)
    func(image_ids, 0.95)
    func(image_ids, 0.9)
    func(image_ids, 0.8)
    func(image_ids, 0.7)
    func(image_ids, 0.6)
    func(image_ids, 0.5)


summary("data/track/annotations/cut_4000x4000/cut_4000x4000_all-check.json")
summary("data/track/annotations/cut_4000x4000_overlap_70/cut_4000x4000_all-check.json")
summary("data/track/annotations/overlap_70_all_category/overlap_70_all_category-check.json")
# summary("data/track/annotations/ori_instance_all_category.json")
# summary("data/track/annotations/ori_instance_all.json")
'''
-------------------------------- image_size= 7393 --------------------------------
-------------------------------- category_id --------------------------------
                 id  image_id    bbox  iscrowd  ignore    area  segmentation
category_id                                                                 
3            107703    107703  107703   107703  107703  107703        107703
4             95710     95710   95710    95710   95710   95710         95710
2             93641     93641   93641    93641   93641   93641         93641
1             18965     18965   18965    18965   18965   18965         18965
-------------------------------- image_id --------------------------------
                                                     id  ...  segmentation
image_id                                                 ...              
09_Electronic_Market/IMG_09_24__007000_003500_0...  907  ...           907
08_Dongmen_Street/IMG_08_24__010500_003500_0145...  806  ...           806
08_Dongmen_Street/IMG_08_07__010500_003500_0145...  790  ...           790
08_Dongmen_Street/IMG_08_08__010500_003500_0145...  787  ...           787
08_Dongmen_Street/IMG_08_11__010500_003500_0145...  786  ...           786

[5 rows x 7 columns]
----------------------------------------------------------------
thr 128.0
> 128.0 size  603
< 128.0 size  6790
----------------------------------------------------------------
percent,  0.95
thr 184
> 184 size  370
< 184 size  7023
----------------------------------------------------------------
percent,  0.9
thr 111
> 111 size  740
< 111 size  6653
----------------------------------------------------------------
percent,  0.8
thr 58
> 58 size  1485
< 58 size  5908
----------------------------------------------------------------
percent,  0.7
thr 35
> 35 size  2249
< 35 size  5144
----------------------------------------------------------------
percent,  0.6
thr 21
> 21 size  2983
< 21 size  4410
----------------------------------------------------------------
percent,  0.5
thr 13
> 13 size  3699
< 13 size  3694
-------------------------------- image_size= 7978 --------------------------------
-------------------------------- category_id --------------------------------
                 id  image_id    bbox  iscrowd  ignore    area  segmentation
category_id                                                                 
3            109852    109852  109852   109852  109852  109852        109852
4            104874    104874  104874   104874  104874  104874        104874
2            103314    103314  103314   103314  103314  103314        103314
1             21850     21850   21850    21850   21850   21850         21850
-------------------------------- image_id --------------------------------
                                                     id  ...  segmentation
image_id                                                 ...              
09_Electronic_Market/IMG_09_24__007000_003500_0...  914  ...           914
08_Dongmen_Street/IMG_08_24__010500_003500_0145...  821  ...           821
08_Dongmen_Street/IMG_08_11__010500_003500_0145...  804  ...           804
08_Dongmen_Street/IMG_08_07__010500_003500_0145...  803  ...           803
08_Dongmen_Street/IMG_08_08__010500_003500_0145...  794  ...           794

[5 rows x 7 columns]
----------------------------------------------------------------
thr 128.0
> 128.0 size  648
< 128.0 size  7330
----------------------------------------------------------------
percent,  0.95
thr 179
> 179 size  401
< 179 size  7577
----------------------------------------------------------------
percent,  0.9
thr 112
> 112 size  803
< 112 size  7175
----------------------------------------------------------------
percent,  0.8
thr 58
> 58 size  1600
< 58 size  6378
----------------------------------------------------------------
percent,  0.7
thr 36
> 36 size  2398
< 36 size  5580
----------------------------------------------------------------
percent,  0.6
thr 21
> 21 size  3212
< 21 size  4766
----------------------------------------------------------------
percent,  0.5
thr 12
> 12 size  4200
< 12 size  3778
-------------------------------- image_size= 8449 --------------------------------
-------------------------------- category_id --------------------------------
                 id  image_id    bbox  iscrowd  ignore    area  segmentation
category_id                                                                 
4            129935    129935  129935   129935  129935  129935        129935
3            109852    109852  109852   109852  109852  109852        109852
2            103314    103314  103314   103314  103314  103314        103314
1             25570     25570   25570    25570   25570   25570         25570
-------------------------------- image_id --------------------------------
                                                     id  ...  segmentation
image_id                                                 ...              
09_Electronic_Market/IMG_09_24__007000_003500_0...  940  ...           940
08_Dongmen_Street/IMG_08_07__010500_003500_0145...  865  ...           865
08_Dongmen_Street/IMG_08_24__010500_003500_0145...  856  ...           856
09_Electronic_Market/IMG_09_05__007000_003500_0...  833  ...           833
08_Dongmen_Street/IMG_08_11__010500_003500_0145...  829  ...           829

[5 rows x 7 columns]
----------------------------------------------------------------
thr 128.0
> 128.0 size  723
< 128.0 size  7726
----------------------------------------------------------------
percent,  0.95
thr 186
> 186 size  424
< 186 size  8025
----------------------------------------------------------------
percent,  0.9
thr 115
> 115 size  854
< 115 size  7595
----------------------------------------------------------------
percent,  0.8
thr 59
> 59 size  1703
< 59 size  6746
----------------------------------------------------------------
percent,  0.7
thr 35
> 35 size  2560
< 35 size  5889
----------------------------------------------------------------
percent,  0.6
thr 20
> 20 size  3417
< 20 size  5032
----------------------------------------------------------------
percent,  0.5
thr 12
> 12 size  4362
< 12 size  4087

Process finished with exit code 0

'''
