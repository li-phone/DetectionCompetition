import os
import glob
import sys
import argparse


def clear(dir, idx=12):
    assert isinstance(idx, int) and idx >= 0
    epochs = glob.glob(os.path.join(dir, 'epoch_*.pth'))
    for e in epochs:
        if os.path.basename(e) != 'epoch_{}.pth'.format(idx):
            latest_pth = os.path.realpath(os.path.join(dir, 'latest.pth'))
            if os.path.basename(e) != os.path.basename(latest_pth):
                print('remove {} ok!'.format(e))
                os.remove(e)


def parse_args():
    parser = argparse.ArgumentParser(description='clear')
    parser.add_argument('dir', help='dir')
    parser.add_argument('--idx', type=int, default=12, help='file prefix')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    clear(args.dir, args.idx)


if __name__ == '__main__':
    main()
