import itertools
import traceback

import numpy as np
import mmcv
import torch
from PIL.Image import Image
from torchvision.io import read_image
import pycocotools.mask as maskUtils
from mmdet.core import mask2ndarray, BitmapMasks
from mmdet.datasets import build_dataset
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms as T
import os
import os.path
from PIL import Image, ImageColor, ImageDraw, ImageFont
import warnings
from typing import List, Optional, Tuple, Union

COLORS = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
          (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
          (0, 0, 192), (250, 170, 30)]
DEVICE = 'cpu'


def draw_bounding_boxes_new(
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
        fill: Optional[bool] = False,
        width: int = 1,
        font: Optional[str] = None,
        font_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If fill is True, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw: Image = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            txt_color = tuple(int(x * 0.8) for x in color)
            margin = 0
            position = (bbox[0] + margin, bbox[1] + margin)
            text_bbox = draw.textbbox(position, label, font=txt_font)
            draw.rectangle(text_bbox, fill="black")
            draw.text(position,
                      label,
                      fill=txt_color,
                      font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


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


def draw_bbox_masks(cfg, dataset=None, save_dir=str, split=None, outputs=None, show_score_thr=0):
    mmcv.mkdir_or_exist(save_dir)

    try:
        if outputs is None and split is not None:
            if split == 'val':
                try:
                    img_filepath = cfg.data.val.img_prefix
                except:
                    print(f'val set does not exist in dataset')
            dataset = build_dataset(cfg.data[split])
            img_filepath = cfg.data[split].img_prefix
            print(f"\nDrawing ground truth bboxes and masks over {len(dataset)} images..........")
            progress_bar = mmcv.ProgressBar(len(dataset))
            for idx in range(len(dataset)):
                image_name = dataset.data_infos[idx]['filename']
                image_path = os.path.join(img_filepath, image_name)
                img = read_image(image_path)
                bboxes = torch.from_numpy(dataset.get_ann_info(idx)['bboxes']).to(device=DEVICE, dtype=torch.float32)
                img_h = dataset.data_infos[idx]['height']
                img_w = dataset.data_infos[idx]['width']
                masks = dataset.get_ann_info(idx).get('masks')
                if any(masks):
                    masks = BitmapMasks(
                        [poly2mask(mask, img_h, img_w) for mask in masks], img_h, img_w)
                    masks = mask2ndarray(masks)
                    masks = torch.from_numpy(masks).to(device=DEVICE, dtype=torch.bool)
                    mask_colors = ["blue"] * masks.shape[0]
                    img = draw_segmentation_masks(img, masks, alpha=0.5, colors=mask_colors)
                    # img = T.ToPILImage()(img)
                    # save_name = dataset.data_infos[idx]['filename'].replace('.jpg', '_mask.png')
                    # save_path = os.path.join(save_dir, save_name)
                    # img.save(save_path)
                bbox_colors = ["blue"] * bboxes.shape[0]
                img = draw_bounding_boxes_new(img, bboxes, width=5, colors=bbox_colors)
                img = T.ToPILImage()(img)
                save_name = dataset.data_infos[idx]['filename'].replace('.jpg', '_mask.png')
                save_path = os.path.join(save_dir, save_name)
                img.save(save_path)

                progress_bar.update()
        elif outputs is not None:
            assert split is None, \
                f'split needs to be None for drawing detected bboxes and masks'
            assert dataset is not None, \
                f'dataset cannot be None for drawing detected bboxes and masks'
            img_filepath = cfg.data.test.img_prefix
            class_names = dataset.CLASSES
            if isinstance(outputs[0], tuple):
                print(f"\nDrawing detected masks over {len(outputs)} images..........")
                progress_bar = mmcv.ProgressBar(len(outputs))
                for idx, output in enumerate(outputs):
                    image_name = dataset.data_infos[idx]['filename']
                    image_path = os.path.join(img_filepath, image_name)
                    img = read_image(image_path)
                    img_h = dataset.data_infos[idx]['height']
                    img_w = dataset.data_infos[idx]['width']
                    # bboxes = torch.from_numpy(output[0][:, :4]).to(device=DEVICE, dtype=torch.float32)
                    bbox_results, segm_results = output
                    for i, (bbox_result, mask_results) in enumerate(zip(bbox_results, segm_results)):
                        scores = torch.from_numpy(bbox_result[:, 4]).to(device=DEVICE, dtype=torch.float32)
                        bboxes = torch.from_numpy(bbox_result[:, :4]).to(device=DEVICE, dtype=torch.float32)
                        labels = [class_names[i]] * bboxes.shape[0]
                        masks = BitmapMasks(
                            [maskUtils.decode(mask) for mask in mask_results], img_h, img_w)
                        masks = mask2ndarray(masks)
                        masks = torch.from_numpy(masks).to(device=DEVICE, dtype=torch.bool)
                        if show_score_thr > 0:
                            assert bboxes is not None and bbox_result.shape[1] == 5
                            inds = scores > show_score_thr
                            bboxes = bboxes[inds, :]
                            masks = masks[inds, :]
                            scores = scores[inds]
                            inds_list = inds.tolist()
                            labels = [labels[i] for i in range(len(labels)) if inds_list[i]]
                        labels = [f'{label}|{scores[j]:.4f}' for j, label in enumerate(labels)]
                        if len(class_names) == 1:
                            y_cycle = itertools.cycle(COLORS)
                            mask_colors = []
                            for j in range(masks.shape[0]):
                                mask_colors.append(next(y_cycle))
                        else:
                            mask_colors = [dataset.PALETTE[i]] * masks.shape[0]
                        bbox_colors = [(240, 10, 10)] * bboxes.shape[0]
                        img = draw_segmentation_masks(img, masks, alpha=0.5, colors=mask_colors)
                        img = draw_bounding_boxes_new(img, bboxes, width=5, labels=labels, colors=bbox_colors,
                                                      font="/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf",
                                                      font_size=10)
                    img = T.ToPILImage()(img)
                    save_name = dataset.data_infos[idx]['filename'].replace('.jpg', '_mask.png')
                    save_path = os.path.join(save_dir, save_name)
                    img.save(save_path)

                    progress_bar.update()

            elif isinstance(outputs[0], list):
                print(f"\nDrawing detected bounding boxes over {len(outputs)} images..........")
                progress_bar = mmcv.ProgressBar(len(outputs))
                for idx, output in enumerate(outputs):
                    image_name = dataset.data_infos[idx]['filename']
                    image_path = os.path.join(img_filepath, image_name)
                    img = read_image(image_path)
                    for i, bbox_results in enumerate(output):
                        scores = torch.from_numpy(bbox_results[:, 4]).to(device=DEVICE, dtype=torch.float32)
                        bboxes = torch.from_numpy(bbox_results[:, :4]).to(device=DEVICE, dtype=torch.float32)
                        labels = [class_names[i]] * bboxes.shape[0]
                        bbox_colors = [(240, 10, 10)] * bboxes.shape[0]
                        # colors = [dataset.PALETTE[i]] * bboxes.shape[0]
                        if show_score_thr > 0:
                            assert bboxes is not None and bbox_results.shape[1] == 5
                            inds = scores > show_score_thr
                            bboxes = bboxes[inds, :]
                            scores = scores[inds]
                            inds_list = inds.tolist()
                            labels = [labels[i] for i in range(len(labels)) if inds_list[i]]
                        labels = [f'{label}|{scores[j]:.4f}' for j, label in enumerate(labels)]
                        img = draw_bounding_boxes_new(img, bboxes, width=5, labels=labels, colors=bbox_colors,
                                                      font="/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf",
                                                      font_size=15)
                    img = T.ToPILImage()(img)
                    save_name = dataset.data_infos[idx]['filename'].replace('.jpg', '_bbox.png')
                    save_path = os.path.join(save_dir, save_name)
                    img.save(save_path)

                    progress_bar.update()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("Both output and split can't be None")
