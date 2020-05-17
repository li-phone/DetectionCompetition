import glob
import os
import sys
import argparse
import shutil
from tqdm import tqdm


def rename_file(parent_dir, prefix=None):
    paths = glob.glob(parent_dir + r"/*")
    for path in tqdm(paths):
        if prefix is None:
            new_path = os.path.basename(parent_dir) + "_" + os.path.basename(path)
        else:
            new_path = prefix + "_" + os.path.basename(path)
        new_path = os.path.join(parent_dir, new_path)
        if os.path.exists(path):
            shutil.move(path, new_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Rename a directory')
    parser.add_argument('dir', help='directory')
    parser.add_argument('prefix', help='file prefix')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    rename_file(args.dir, args.prefix)


if __name__ == '__main__':
    main()
