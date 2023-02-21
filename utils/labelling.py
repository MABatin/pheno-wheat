import glob
import json

import cv2
import imagesize
import mmcv
import numpy as np
from mmcv import Config
from mmdet.apis import inference_detector, init_detector


def xyxy2xywh(bbox):
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def polygonFromMask(maskedArr):
    area = float((maskedArr > 0.0).sum())
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    # additionally used RDP algorithm for contour approximation to reduce number of vertices
    contours, _ = cv2.findContours(maskedArr.astype(np.ubyte), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    eps = 0.01  # 0.01 is the optimal value
    peri = cv2.arcLength(c, True)
    approxs = cv2.approxPolyDP(c, eps * peri, True)
    segmentation = []
    # valid_poly = 0
    # for contour in contours:
    #     # Valid polygons have >= 6 coordinates (3 points)
    #     if contour.size >= 6:
    #         segmentation.append(contour.astype(float).flatten().tolist())
    #         valid_poly += 1
    # if valid_poly == 0:
    #     raise ValueError
    segmentation.append(approxs.astype(float).flatten().tolist())
    return segmentation, area


def result2json():
    config = 'configs/models/cascade_mask_rcnn_mod_250e_coco.py'
    checkpoint_file = 'checkpoints/Trained/SPIKE_small/cascade_mask_rcnn_mod_epoch_250.pth'
    img_dir = 'data/SPIKE_main_imgonly/val'
    cfg = Config.fromfile(config)
    model = init_detector(cfg, checkpoint_file)
    images = glob.glob(img_dir + '/*.jpg', recursive=True)
    json_results = dict()
    json_results['categories'] = []
    json_results['images'] = []
    json_results['annotations'] = []
    ann_id = 0
    total_points = 0
    for idx in mmcv.track_iter_progress(range(len(images))):
        result = inference_detector(model, images[idx])
        # print(f"results memory usage: {(asizeof(result)/ 1024 / 1024 / 1024): .4f}GB")
        image = dict()
        image_file = images[idx]
        image_name = image_file.split('\\')[-1]
        width, height = imagesize.get(image_file)
        image['id'] = idx
        image['file_name'] = image_name
        image['height'] = height
        image['width'] = width
        json_results['images'].append(image)

        bbox_result, segm_result = result
        for i in range(len(segm_result)):
            for j, mask in enumerate(segm_result[i]):
                annotation = dict()
                annotation['id'] = ann_id
                annotation['image_id'] = idx
                annotation['category_id'] = i + 1
                annotation['bbox'] = xyxy2xywh(bbox_result[i][j][:-1])
                segm, area = polygonFromMask(mask)
                annotation['area'] = area
                annotation['segmentation'] = segm
                num_points = 0
                for k in range(len(segm)):
                    num_points += len(segm[k])
                annotation['num_points'] = num_points
                annotation['iscrowd'] = 0
                json_results['annotations'].append(annotation)
                ann_id += 1
                total_points += num_points

    for i in range(len(cfg.data.test.classes)):
        cat = dict()
        cat['id'] = i + 1
        cat['name'] = cfg.data.test.classes[i]
        json_results['categories'].append(cat)

    json_results['total_points'] = total_points
    with open('data/SPIKE_main_imgonly/val.json', 'w') as f:
        json.dump(json_results, f)


if __name__ == '__main__':
    result2json()
