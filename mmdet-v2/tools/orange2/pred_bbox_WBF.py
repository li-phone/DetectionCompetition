#encoding=utf-8
"""
Project : project
File    : pred_bbox_WBF #todo 对预测出的结果(类别，score,x1,y1,x2,y2)进行WBF框处理;备注 bbox需归一化norx1,nory1,norx2,nory2
Author  : PT
CreateTime = 2021-07-02 16:03
"""
from ensemble_boxes import *
import os
import cv2 as cv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

calss_dict={'HLB':0, 'health':1, 'ill':2}
category_dict={0:'HLB', 1:'health', 2:'ill'}

def x1y1x2y2_to_normal(boxes_list,size):
    h = size[0]
    w = size[1]
    boxes_np = np.array(boxes_list)
    boxes_np[:,::2] = boxes_np[:,::2]/w
    boxes_np[:,1::2] = boxes_np[:,1::2]/h
    return boxes_np.tolist()

def mutli_x1y1x2y2_to_normal(boxes_list,size):
    h = size[0]
    w = size[1]
    boxe_list=[]
    for boxes_l in boxes_list:
        boxes_np = np.array(boxes_l)
        boxes_np[:,::2] = boxes_np[:,::2]/w
        boxes_np[:,1::2] = boxes_np[:,1::2]/h
        boxe_list.append(boxes_np.tolist())
    return boxe_list

def normal_to_x1y1x2y2(boxes_list,size):
    h = size[0]
    w = size[1]
    boxes_np = np.array(boxes_list)
    boxes_np[:,::2] = boxes_np[:,::2]*w
    boxes_np[:,1::2] = boxes_np[:,1::2]*h

    return boxes_np.tolist()

#todo 单个文件中的bboxes融合
def boxes_wbf(in_f,in_image,out_f):
    img = cv.imread(in_image)
    size= img.shape[:2]   #todo h=size[0] ; w=size[1]
    labels_list=[]
    boxes_list=[]
    for line in open(in_f,"r").readlines():
        split_line=line.strip().split(" ")
        labels_list.append(calss_dict[split_line[0]])
        boxes_list.append([float(split_line[1]),float(split_line[2]),float(split_line[3]),float(split_line[4])])

    scores_list = [1] * len(labels_list)

    boxes_mor_list=x1y1x2y2_to_normal(boxes_list,size)  #todo 将坐标进行归一化，满足wbf的要求
    if len(boxes_list) !=1:
        iou_thr = 0.5
        skip_box_thr = 0.0001
        sigma = 0.1
        boxes_t, scores_t, labels_t = weighted_boxes_fusion([boxes_mor_list], [scores_list], [labels_list],weights=None,iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes_list_t = normal_to_x1y1x2y2(boxes_t,size)
        outf = open(out_f,"a")
        for boxe,score,label in zip(boxes_list_t,scores_t,labels_t):
            category=category_dict[label]
            string = category + " " + str(boxe[0])+" " + str(boxe[1])\
                     + " " + str(boxe[2])+" " + str(boxe[3])+"\n"  #todo 对数据前处理的wbf
            outf.write(string)
        outf.close()
    else:
        shutil.copy(in_f,out_f)

#todo 多文件中的bboxes融合
def mutli_boxes_wbf(in_f_list,in_image,out_f):
    img = cv.imread(in_image)
    size= img.shape[:2]   #todo h=size[0] ; w=size[1]
    labels_list=[]
    boxes_list=[]
    scores_list=[]
    for in_f in in_f_list:
        labels_list_t=[]
        boxes_list_t=[]
        scores_list_t=[]
        for line in open(in_f,"r").readlines():
            split_line=line.strip().split(" ")
            labels_list_t.append(calss_dict[split_line[0]])
            scores_list_t.append(float(split_line[1]))
            boxes_list_t.append([float(split_line[2]),float(split_line[3]),float(split_line[4]),float(split_line[5])])
        labels_list.append(labels_list_t)
        boxes_list.append(boxes_list_t)
        scores_list.append(scores_list_t)

    boxes_mor_list=mutli_x1y1x2y2_to_normal(boxes_list,size)  #todo 将坐标进行归一化，满足wbf的要求

    if len(boxes_list) !=1:
        iou_thr = 0.45
        skip_box_thr = 0.0001
        sigma = 0.1
        #boxes_t, scores_t, labels_t = nms(boxes_mor_list, scores_list, labels_list, weights=None,iou_thr=iou_thr)
        #boxes_t, scores_t, labels_t = soft_nms(boxes_mor_list, scores_list, labels_list, weights=None,iou_thr=iou_thr,  sigma=sigma, thresh=skip_box_thr)
        boxes_t, scores_t, labels_t = non_maximum_weighted(boxes_mor_list, scores_list, labels_list, weights=None,
                                                             iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # boxes_t, scores_t, labels_t = weighted_boxes_fusion(boxes_mor_list, scores_list, labels_list,weights=None,
        #                                                     iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes_list_t = normal_to_x1y1x2y2(boxes_t,size)
        outf = open(out_f,"a")
        for boxe,score,label in zip(boxes_list_t,scores_t,labels_t):
            category=category_dict[label]
            string = category + " " +str(round(score,4)) +" "+ str(int(round(boxe[0])))+" " + str(int(round(boxe[1])))\
                     + " " + str(int(round(boxe[2])))+" " + str(int(round(boxe[3])))+"\n"  #todo 对数据前处理的wbf
            outf.write(string)
        outf.close()
    else:
        shutil.copy(in_f,out_f)

if __name__=="__main__":
    inputfile1 = "/home/mnt_abc004-data/mq/Project/yolov5-master/runs/detect-planet-diseases/exp-ensemble_boxes-non_maximum_weighted-13-11-9-4/labels-15"
    inputfile2 = "/home/mnt_abc004-data/mq/Project/yolov5-master/runs/detect-planet-diseases/exp-ensemble_boxes-non_maximum_weighted-13-11-9-4/labels-16"
    inputfile3 = "/home/mnt_abc004-data/mq/Project/yolov5-master/runs/detect-planet-diseases/exp-ensemble_boxes-non_maximum_weighted-13-11-9-4/labels-ori"
    #inputfile4 = "/home/mnt_abc004-data/mq/Project/yolov5-master/runs/detect-planet-diseases/exp-WBF-11-9-4-Ensemble/labels-E"

    inputimage = "/home/mnt_abc004-data/mq/dataset/plant_diseases/test/images"
    outputfile = "/home/mnt_abc004-data/mq/Project/yolov5-master/runs/detect-planet-diseases/exp-ensemble_boxes-non_maximum_weighted-13-11-9-4/labels-non_maximum_weighted"

    names = os.listdir(inputimage)
    # for name in names:  #todo 单一文件的加权处理
    #     in_file = os.path.join(inputfile1,name.split(".")[0]+".txt")
    #     in_image = os.path.join(inputimage, name)
    #     out_file = os.path.join(outputfile,name.split(".")[0]+".txt")
    #     boxes_wbf(in_file, in_image, out_file)

    for name in names:  #todo 多文件的加权处理
        in_file1 = os.path.join(inputfile1, name.split(".")[0] + ".txt")
        in_file2 = os.path.join(inputfile2, name.split(".")[0] + ".txt")
        in_file3 = os.path.join(inputfile3, name.split(".")[0] + ".txt")
        #in_file4 = os.path.join(inputfile4, name.split(".")[0] + ".txt")
        in_image = os.path.join(inputimage, name)
        out_file = os.path.join(outputfile, name.split(".")[0] + ".txt")
        mutli_boxes_wbf([in_file1,in_file2,in_file3], in_image, out_file)

    print("###finshed!!")