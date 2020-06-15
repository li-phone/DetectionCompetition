import argparse

from utils.datasets import *
from utils.utils import *


def init_detector(weights, device='', half=False):
    # Initialize
    device = torch_utils.select_device(device)

    # Load model
    model = torch.load(weights, map_location=device)['model']
    model.to(device).eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    return model, modelc if classify else None


def inference_detector(model, images, img_size=640, device='',
                       conf_thres=0.4, iou_thres=0.5, classes=None,
                       agnostic_nms=False, augment=False, half=False):
    if not isinstance(images, list):
        images = [images]

    # Initialize
    device = torch_utils.select_device(device)

    classify = False

    # Run inference
    results = []
    for i, image in enumerate(images):
        img, im0s = test_pipeline(image, img_size)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img, augment=augment)[0]

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   fast=True, classes=classes, agnostic=agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        results.append(pred)

    return tuple(results)


if __name__ == '__main__':
    import time

    t0 = time.time()
    model, modelc = init_detector('weights/best.pt', device='cpu')
    images = [
        'inference/images/31fec8f5956713922c9dcfac9a89edf.jpg',
        'inference/images/9d0ac700d8dafd5c568fd3d78224ffb.jpg',
    ]
    result = inference_detector(model, images, device='cpu')
    print('Done. (%.3fs)' % (time.time() - t0))
    print(result)
