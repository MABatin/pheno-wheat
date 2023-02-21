import json

import cv2, os, mmcv, json
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.datasets.api_wrappers import COCO

# reading the image in grayscale mode


def count_dots(path):
    gray = cv2.imread(path, 0)

    # threshold
    th, threshed = cv2.threshold(gray, 100, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    # filter by area
    s1 = 2
    s2 = 100
    xcnts = []

    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            xcnts.append(cnt)

    # printing output
    # print("\nImage: {}, Dots number: {}".format(image_name, len(xcnts)))
    return len(xcnts)


def check_count(split_key=None, model_config=None):
    dataset_config = 'Wheat/configs/datasets/spikelet_instance.py'
    cfg = Config.fromfile(dataset_config)
    # cfg.data_root = 'Wheat/data/Spikelet_small_v1/'

    if model_config is not None:
        directory = 'Wheat/results/' + cfg.data_root.split('/')[-2] + '/' + model_config + '/segm'
        dataset = build_dataset(cfg.data.test)

        results = []
        error = []
        total_oc = 0
        total_uc = 0

        for idx in range(len(dataset)):
            info = dict()
            filename = dataset.get_ann_info(idx)['seg_map'].replace('.png', '_segm.png')
            img_path = os.path.join(directory, filename)

            num_gt = dataset.get_ann_info(idx)['bboxes'].shape[0]
            num_count = count_dots(img_path)

            info['img_id'] = idx
            info['img_name'] = filename
            info['gt_count'] = num_gt
            info['det_count'] = num_count

            if num_gt != num_count:
                oc = max(0, (num_count-num_gt))
                uc = max(0, (num_gt-num_count))
                total_oc += oc
                total_uc += uc
                info['OC'] = oc
                info['UC'] = uc
            error.append(abs(num_count - num_gt))

            results.append(info)

        mse = np.square(error).mean(axis=None)
        results.append(dict(total_OC=total_oc, total_UC=total_uc, mSE=mse))

        json_path = 'Wheat/results/' + cfg.data_root.split('/')[-2] + '/' + model_config
        json_name = json_path + '/count.json'

        with open(json_name, 'w') as f:
            json.dump(results, f)

        print(results)

    elif model_config is None and split_key is not None:
        if split_key == 'train':
            dataset = build_dataset(cfg.data.train)
        elif split_key == 'test':
            dataset = build_dataset(cfg.data.test)
        directory = cfg.data_root + '/segm/' + split_key

        results = []
        diff = []
        flag = True

        for idx in range(len(dataset)):
            info = dict()
            filename = dataset.get_ann_info(idx)['seg_map'].replace('.png', '_segm.png')
            img_path = os.path.join(directory, filename)

            num_gt = dataset.get_ann_info(idx)['bboxes'].shape[0]
            num_count = count_dots(img_path)

            info['img_id'] = idx
            info['img_name'] = filename
            info['gt'] = num_gt
            info['count'] = num_count
            if num_gt != num_count:
                flag = False
                diff.append([idx, filename])

            results.append(info)
        results.append(dict(all_same=flag))
        results.append(dict(not_same_num=len(diff)))
        results.append(dict(not_same=diff))

        json_path = cfg.data_root
        json_name = json_path + split_key + '_count.json'

        with open(json_name, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    #splits = ['train', 'test']
    #for split in splits:
    #    check_count(split_key=split)
    check_count(model_config='fcn_unet_s5_d16_80k_b2_n')


# coco = COCO(cfg.data.test.ann_file)
# img_id = data_info[idx]['id']
# ann_ids = coco.get_ann_ids(img_ids=[img_id])
# ann_info = coco.load_anns(ann_ids)