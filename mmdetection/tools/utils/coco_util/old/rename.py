# MIT License
#
# Copyright(c) [2019] [liphone/lifeng] [email: 974122407@qq.com]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this softwareand associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright noticeand this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import glob
import os
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


def _test_rename_file():
    rename_file('G:/data/pascalvoc/coco_pascalvoc/VOC2007/JPEGImages', "VOCtrainval_06-Nov-2007")
    pass


if __name__ == '__main__':
    _test_rename_file()
    pass
