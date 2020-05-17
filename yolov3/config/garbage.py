classes = 44
train = "/home/liphone/undone-work/data/detection/garbage_huawei/yolo/trainval.txt"
valid = "/home/liphone/undone-work/data/detection/garbage_huawei/yolo/trainval.txt"
label_list = "/home/liphone/undone-work/data/detection/garbage_huawei/yolo/label_list.txt"
work_dirs = './work_dirs/garbage/yolov3/'

# train config
epochs = 60
lr = 1e-4
batch_size = 4

gradient_accumulations = 2
model_def = "config/yolov3-44.cfg"
pretrained_weights = 'weights/darknet53.conv.74'

n_cpu = 8
img_size = 416

checkpoint_interval = 1
evaluation_interval = 10
compute_map = True

multiscale_training = True
