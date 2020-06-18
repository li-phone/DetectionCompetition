# get model parameters
import torch


def get_parameter_number(net):
    total = dict(bk=0, sub=0, dfn=0)
    trainable = dict(bk=0, sub=0, dfn=0)
    for k, v in net.items():
        if 'backbone.fc' in k:
            total['dfn'] += v.numel()
            if v.requires_grad:
                trainable['dfn'] += v.numel()
        elif 'backbone.' in k:
            total['bk'] += v.numel()
            if v.requires_grad:
                trainable['bk'] += v.numel()
        else:
            total['sub'] += v.numel()
            if v.requires_grad:
                trainable['sub'] += v.numel()
    return {'Total': total, 'Trainable': trainable}


def main():
    m = torch.load(
        '/home/liphone/undone-work/DetCompetition/tmp/yolov5/weights/last.pth', map_location='cpu')
    print(get_parameter_number(m['state_dict']))


if __name__ == '__main__':
    main()