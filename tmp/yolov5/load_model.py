import torch

m = torch.load('weights/yolov5x.pt')
print(m)
m['epoch'] = -1
torch.save(m, 'weights/yolov5x.pt')
