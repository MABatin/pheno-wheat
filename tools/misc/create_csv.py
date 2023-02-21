import argparse

import mmcv
import pandas as pd
import os
import os.path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create csv annotation file from SPIKE dataset'
    )
    parser.add_argument('file_dir', help='Directory of dataset')
    parser.add_argument('dir_type', help='train/val/test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root = args.file_dir
    directory = args.dir_type
    filepath = os.path.join(root, directory)

    concated_bbox = []
    concated_label = []

    img_filenames = [name for name in os.listdir(filepath)
                     if os.path.splitext(name)[-1] == '.jpg']
    box_filenames = [name for name in os.listdir(filepath)
                     if os.path.splitext(os.path.splitext(name)[0])[-1] == '.bboxes']
    label_filenames = [name for name in os.listdir(filepath)
                       if os.path.splitext(os.path.splitext(name)[0])[-1] == '.labels']

    for idx, box in enumerate(box_filenames):
        box_df = pd.read_csv(filepath + '/' + box, sep='\t', header=None)
        label_df = pd.read_csv(filepath + '/' + label_filenames[idx], sep='\t', header=None)

        box_df.insert(0, 'image_id', img_filenames[idx])
        concated_bbox.append(box_df)
        concated_label.append(label_df)

    concated_bbox = pd.concat(concated_bbox, axis=0)
    concated_bbox.columns = ['image_id', 'XMin', 'YMin', 'XMax', 'YMax']
    concated_label = pd.concat(concated_label, axis=0)
    concated_label.columns = ['Classname']

    df = pd.concat([concated_bbox, concated_label], axis=1)
    df.loc[df['Classname'] == 'S', 'Classname'] = 'Spike'

    ann_path = os.path.join(root, 'annotations')
    mmcv.mkdir_or_exist(ann_path)
    print(f'Saving to {ann_path}/{directory}.csv')
    df.to_csv(ann_path + '/' + directory + '.csv')


if __name__ == '__main__':
    main()
