import json
import json


def check_json_with_error_msg(pred_json, num_classes=10):
    '''
    Args:
        pred_json (str): Json path
        num_classes (int): number of foreground categories
    Returns:
        Message (str)
    Example:
        msg = check_json_with_error_msg('./submittion.json')
        print(msg)
    '''
    if not pred_json.endswith('.json'):
        return "the prediction file should ends with .json"
    with open(pred_json) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return "the prediction data should be a dict"
    if not 'images' in data:
        return "missing key \"images\""
    if not 'annotations' in data:
        return "missing key \"annotations\""
    images = data['images']
    annotations = data['annotations']
    if not isinstance(images, (list, tuple)):
        return "\"images\" format error"
    if not isinstance(annotations, (list, tuple)):
        return "\"annotations\" format error"
    for image in images:
        if not 'file_name' in image:
            return "missing key \"file_name\" in \"images\""
        if not 'id' in image:
            return "missing key \"id\" in \"images\""
    for annotation in annotations:
        if not 'image_id' in annotation:
            return "missing key \"image_id\" in \"annotations\""
        if not 'category_id' in annotation:
            return "missing key \"category_id\" in \"annotations\""
        if not 'bbox' in annotation:
            return "missing key \"bbox\" in \"annotations\""
        if not 'score' in annotation:
            return "missing key \"score\" in \"annotations\""
        if not isinstance(annotation['bbox'], (tuple, list)):
            return "bbox format error"
        if len(annotation['bbox']) == 0:
            return "empty bbox"
        if annotation['category_id'] > num_classes or annotation['category_id'] < 0:
            return "category_id out of range"
    return ""


def do_filter(file_path, save_path):
    with open(file_path) as fp:
        results = json.load(fp)
    for r in results['annotations']:
        b = r['bbox']
        b[2] = b[2] - b[0]
        b[3] = b[3] - b[1]
    with open(save_path, 'w') as fp:
        json.dump(results, fp)


if __name__ == '__main__':
    error_msg = check_json_with_error_msg(
        '/home/liphone/undone-work/uselessNet/code/mmdetection/work_dirs/alcohol/cascade_rcnn_r50_fpn_1x_freeze/filter_cascade_rcnn_r50_fpn_1x_alcohol_latest_train_submit.json')
    print('error msg:', error_msg)
    do_filter(
        '/home/liphone/undone-work/uselessNet/code/mmdetection/work_dirs/alcohol/cascade_rcnn_r50_fpn_1x_freeze/cascade_rcnn_r50_fpn_1x_alcohol_latest_train_submit.json',
        '/home/liphone/undone-work/uselessNet/code/mmdetection/work_dirs/alcohol/cascade_rcnn_r50_fpn_1x_freeze/filter_cascade_rcnn_r50_fpn_1x_alcohol_latest_train_submit.json'
    )
