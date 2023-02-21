import argparse
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
import torch
from torchvision.io import read_image
import pycocotools.mask as maskUtils
from mmdet.core import mask2ndarray, BitmapMasks
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms as T
import os
import os.path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Draw ground truth as well as detected bbox and masks'
    )
    parser.add_argument('config', help='Dataset config file path')
    parser.add_argument('save_dir', help='Save directory in results/{model name}/bbox(or mask)')
    parser.add_argument('--with_masks', action='store_true', help='Draw segmentation masks if available')
    args = parser.parse_args()
    return args


def poly2mask(mask_ann, img_h, img_w):
    """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decoded bitmap mask of shape (img_h, img_w).
        """

    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def main():
    args = parse_args()
    img_filepath = args.img_dir
    mmcv.mkdir_or_exist(os.path.abspath(args.save_dir))
    save_dir = args.save_dir
    cfg = Config.fromfile(args.config)
    # cfg.data.test.pipeline = cfg.train_pipeline
    dataset = build_dataset(cfg.data.test)
    # img_filepath = cfg.data.test.img_prefix
    progress_bar = mmcv.ProgressBar(len(dataset))

    for idx in range(len(dataset)):
        image_name = dataset.data_infos[idx]['filename']
        image_path = os.path.join(img_filepath, image_name)
        img = read_image(image_path)
        bboxes = torch.from_numpy(dataset.get_ann_info(idx)['bboxes']).to(device='cuda', dtype=torch.float32)
        img_h = dataset.data_infos[idx]['height']
        img_w = dataset.data_infos[idx]['width']
        masks = dataset.get_ann_info(idx).get('masks')
        if any(masks) and args.with_masks:
            masks = BitmapMasks(
                [poly2mask(mask, img_h, img_w) for mask in masks], img_h, img_w)
            masks = mask2ndarray(masks)
            masks = torch.from_numpy(masks).to(device='cuda', dtype=torch.bool)
            img = draw_segmentation_masks(img, masks, alpha=0.3, colors=["blue"] * masks.shape[0])
        img = draw_bounding_boxes(img, bboxes, width=3, colors=["blue"] * bboxes.shape[0])
        img = T.ToPILImage()(img)
        save_name = dataset.data_infos[idx]['filename'].replace('.jpg', '_mask_bbox.jpg')
        save_path = os.path.join(save_dir, save_name)
        img.save(save_path)

        progress_bar.update()


if __name__ == '__main__':
    main()
