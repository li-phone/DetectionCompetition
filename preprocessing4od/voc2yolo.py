from xml.dom.minidom import parse
import os
import shutil
classes = {"holothurian": 0, "echinus": 1, "scallop": 2, "starfish": 3, "waterweeds": 4}


def read_xml(xml_path):
    doc = parse(xml_path)
    x_min = doc.getElementsByTagName('xmin')
    y_min = doc.getElementsByTagName('ymin')
    x_max = doc.getElementsByTagName('xmax')
    y_max = doc.getElementsByTagName('ymax')
    name = doc.getElementsByTagName('name')
    width = int(doc.getElementsByTagName('width')[0].firstChild.data)
    height = int(doc.getElementsByTagName('height')[0].firstChild.data)
    boxes = list()
    for i in range(len(name)):
        x1 = int(x_min[i].firstChild.data)
        y1 = int(y_min[i].firstChild.data)
        x2 = int(x_max[i].firstChild.data)
        y2 = int(y_max[i].firstChild.data)
        cat_name = name[i].firstChild.data
        boxes.append([x1, y1, x2-x1, y2-y1, cat_name])
        # cat_names.add(cat_name)
    return boxes, width*1.0, height*1.0


path = 'F:/under_water/train/train/'
xml_list = os.listdir(path+'box')
cnt = 0
for xml in xml_list:
    # print(xml)
    if cnt % 100 == 0:
        print(f'{cnt}/{len(xml_list)}')
    cnt += 1
    image_boxes, image_width, image_height = read_xml(path + 'box/' + xml)
    if cnt < int(len(xml_list)*0.8):
        f = open(path+'yolo_train/label/'+xml.replace('.xml', '.txt'), 'w')
        shutil.copy(path+'image/'+xml.replace('.xml', '.jpg'), path+'yolo_train/images/'+xml.replace('.xml', '.jpg'))
        for box in image_boxes:
            [x, y, w, h] = box[:4]
            x_center = x + w / 2
            y_center = y + h / 2
            x_center = x_center / image_width
            y_center = y_center / image_height
            w = w / image_width
            h = h / image_height
            x_center = max(0.0, x_center / image_width)
            y_center = max(0.0, y_center / image_height)
            w = min(w / image_width, 1.0)
            h = min(h / image_height, 1.0)
            # print(classes[box[-1]])
            f.write(str(classes[box[-1]])+' '+str(x_center)+' '+str(y_center)+' '+str(w)+' '+str(h)+'\n')
        f.close()
    else:
        f = open(path + 'yolo_val/label/' + xml.replace('.xml', '.txt'), 'w')
        shutil.copy(path + 'image/' + xml.replace('.xml', '.jpg'),
                    path + 'yolo_val/images/' + xml.replace('.xml', '.jpg'))
        for box in image_boxes:
            [x, y, w, h] = box[:4]
            x_center = x + w / 2
            y_center = y + h / 2
            x_center = max(0.0, x_center / image_width)
            y_center = max(0.0, y_center / image_height)
            w = min(w / image_width, 1.0)
            h = min(h / image_height, 1.0)
        f.write(str(classes[box[-1]]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n')
        f.close()
