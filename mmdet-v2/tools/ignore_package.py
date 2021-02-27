import os
import glob
import os.path as osp


def process_one(file, ignore_pkgs):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        has_word = False
        for i, line in enumerate(lines):
            words = line.split()
            if len(words) < 2:
                continue
            if words[0] != 'from' and words[0] != 'import':
                continue
            for pkg in ignore_pkgs:
                if pkg in words[1]:
                    if i >= 1 and i + 1 < len(lines) and 'try:' in lines[i - 1] and 'except:' in lines[i + 1]:
                        continue
                    newline = ' '.join(words)
                    newline = 'try:\n\t' + newline + '\nexcept:\n\tpass\n'
                    lines[i] = newline
                    has_word = True
                    break
        if not has_word:
            return
    with open(file, 'w') as fp:
        fp.writelines(lines)


def process(path, ignore_pkgs):
    file_cnt, dir_cnt = 0, 0
    root = [path]
    while root:
        file = root.pop()
        if osp.isfile(file) and file[-3:] == '.py':
            file_cnt += 1
            process_one(file, ignore_pkgs)
        elif osp.isdir(file):
            dir_cnt += 1
            tmp = glob.glob(osp.join(file, '*'))
            root.extend(tmp)
    print("file count:", file_cnt, ", dir count: ", dir_cnt)


def main():
    PATH = "../../review_code/min_mmdet_v2/mmdet/mmdet"
    PKGS = [
        "terminaltables",
        "pycocotools",
    ]
    process(PATH, PKGS)
    print('ok!')


if __name__ == '__main__':
    main()
