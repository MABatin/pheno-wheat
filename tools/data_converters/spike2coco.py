import argparse
import os.path
import pandas as pd

import mmcv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert spike dataset to coco format with annotations')
    parser.add_argument('file_dir', help='The root path of images')
    parser.add_argument('annotations', type=str, help='csv file name of annotations')
    parser.add_argument(
        'out',
        type=str,
        help='output json file name, save dir is in annotations folder')
    parser.add_argument('--e',
                        type=str,
                        nargs='+',
                        help='The suffix of image to be exclude, e.g. "png", "bmp"')
    args = parser.parse_args()
    return args


def collect_image_infos(path, exclude_extensions=None):
    img_infos = []

    images_generator = mmcv.scandir(dir_path=path,
                                    suffix=('.jpg', '.png', '.bmp'),
                                    recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            image_name = os.path.join(path, image_path)
            img_pillow = Image.open(image_name)
            replace = path + '\\'
            img_info = {
                'filename': image_name.replace(replace, ''),
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)

    return img_infos


def cvt_to_coco_json(img_infos, ann_file):
    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    for idx, value in enumerate(ann_file.Classname.unique()):
        coco['categories'].append(dict(id=idx+1, name=value))
    coco['annotations'] = []
    image_set = []

    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.append(file_name)

        image_id += 1

    annotations = []
    obj_count = 0
    indices = list(range(len(image_set)))
    id_dictionary = dict(zip(image_set, indices))
    for idx, v in ann_file.iterrows():
        x_min, y_min, x_max, y_max = (
            v['XMin'], v['YMin'], v['XMax'], v['YMax'])

        data_anno = dict(
            image_id=id_dictionary[v['image_id']],
            id=obj_count,
            category_id=1,
            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
            area=(x_max - x_min) * (y_max - y_min),
            segmentation=[],
            iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1
    coco['annotations'] = annotations

    return coco


def main():
    args = parse_args()
    assert args.out.endswith('json')  # The output file must be json suffix
    data_root = args.file_dir

    # 1 load image list info
    img_infos = collect_image_infos(os.path.join(args.file_dir, os.path.splitext(args.annotations)[0]), args.e)

    # 2 convert to coco format data
    ann_path = os.path.join(data_root, 'annotations/')
    ann_filename = os.path.join(ann_path, args.annotations)
    annotations = pd.read_csv(ann_filename)
    coco_info = cvt_to_coco_json(img_infos, annotations)

    # 3 dump
    out = os.path.join(ann_path, args.out)
    mmcv.dump(coco_info, out)
    print(f'save json file: {ann_path}')


if __name__ == '__main__':
    main()

