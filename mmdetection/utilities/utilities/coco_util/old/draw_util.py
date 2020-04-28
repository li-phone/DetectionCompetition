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


from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_img(image_path):
    im = Image.open(image_path)
    if im.mode == '1' or im.mode == 'L' or im.mode == 'I' \
            or im.mode == 'F' or im.mode == 'P' or im.mode == 'RGBA' \
            or im.mode == 'CMYK' or im.mode == 'YCbCr':
        im = im.convert('RGB')
    else:
        im = im.convert('RGB')

    return im


def save_img(image, image_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(image_path)


def check_contain_chinese(check_str):
    for c in check_str:
        if '\u4e00' <= c <= '\u9fa5':
            return True
    return False


def check_unicode_len(check_str):
    ch_len = 0
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fa5':
            ch_len += 2
        else:
            ch_len += 1
    return ch_len


def draw_bbox(image, bboxs, labels, label_list, colors):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('simsun.ttc', 24, encoding="uti-8")
    im_width, im_height = image.size
    for label, bbox in zip(labels, bboxs):
        (left, top, right, bottom) = ((bbox[0]), (bbox[1]), (bbox[2]), (bbox[3]))
        if isinstance(label, str):
            idx = len(label_list) - int(label_list.index(label))
        else:
            idx = label
            label = label_list[label]
        bgc = (255 - colors[idx][0], 255 - colors[idx][1], 255 - colors[idx][2])
        character_size = check_unicode_len(label) * 14
        if top - 32 > 0 and left + character_size < im_width:
            pos_x = left
            pos_y = top - 32
        elif top + 32 < im_height and right + character_size < im_width:
            pos_x = right
            pos_y = top
        elif bottom + 32 < im_height and left + character_size < im_width:
            pos_x = left
            pos_y = bottom
        elif top + 32 < im_height and left - character_size > 0:
            pos_x = left - character_size
            pos_y = top
        else:
            pos_x = left
            pos_y = top

        pos_right = pos_x + character_size
        pos_bottom = pos_y + 32

        draw.rectangle(xy=((pos_x, pos_y), (pos_right, pos_bottom)), fill=bgc)
        bgc = (0, 0, 0)
        draw.text((pos_x + 4, pos_y + 4), label, bgc, font=font)
        color = colors[int(label_list.index(label))]
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=3,
            fill=color)

    return image


def get_colors(types_len):
    group_num = types_len ** (1 / 3.)
    group_num = int(group_num + 1)
    group_num = (types_len + group_num) ** (1 / 3.)
    group_num = int(group_num + 1)
    if group_num < 3:
        group_num = 3
    colors = []
    step = int(255 / (group_num - 1))
    rgb_vals = [x for x in range(0, 256, step)]
    for red in rgb_vals:
        for green in rgb_vals:
            for blue in rgb_vals:
                if red == green and red == blue:
                    continue
                colors.append((red, green, blue))
    np.random.shuffle(colors)
    return colors
