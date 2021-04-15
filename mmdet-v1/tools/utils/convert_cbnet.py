import torch
import sys
import os
import argparse


def cvt_cbnet(model_path, K=2):
    f = torch.load(model_path)
    if 'state_dict' in f.keys():
        f = f['state_dict']
    names = list(f.keys())
    prefix = 'db'
    for item in names:
        if 'layer' in item:
            f['second_' + item] = f[item]
            if K == 3:
                f['third_' + item] = f[item]
                prefix = 'tb'
    model_path = model_path.replace('\\', '/')
    save_dir = model_path[:model_path.rfind('/')]
    torch.save(f, os.path.join(save_dir, '{}-{}'.format(prefix, os.path.basename(model_path))))


def parse_args():
    parser = argparse.ArgumentParser(description='cvt_cbnet')
    parser.add_argument('model_path', help='model_path')
    parser.add_argument('--K', type=int, default=2, help='file prefix')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cvt_cbnet(args.model_path, args.K)


if __name__ == '__main__':
    main()
