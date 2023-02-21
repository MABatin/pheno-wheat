# TODO: write a script for the following pipeline: field image -> inference(spike_detection_model) -> crop results -> inference(spikelet_segmentation)
# TODO: -> count(spikelet) -> yield estimation

import argparse
import glob
import mmcv
import os
import torch
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Yield estimation from a given Wheat field image'
    )
    parser.add_argument('img_dir', help='Directory of field images')
    parser.add_argument('output_dir', help='Directory for output images')
    args = parser.parse_args()

    return args


def inference_spike(config, checkpoint, images):
    from mmdet.apis import inference_detector, init_detector
    model = init_detector(config, checkpoint)
    results = inference_detector(model, images)

    return results


def inference_spikelet(config, checkpoint, images):
    from mmseg.apis import inference_segmentor, init_segmentor
    model = init_segmentor(config, checkpoint)
    results = []
    for img in mmcv.track_iter_progress(images):
        result = inference_segmentor(model, img)
        results.append(result)
    # results = inference_segmentor(model, images)

    return results


def crop_det_bbox_mask(images, outputs, save_dir):
    image_folders = []
    for idx in range(len(images)):
        img = mmcv.imread(images[idx], channel_order='rgb')
        img_T = read_image(images[idx])
        img_pil = T.ToPILImage()(img)
        if isinstance(outputs[idx], list):
            bboxes = outputs[idx][0][:-1]
            bboxes = np.squeeze(np.asarray(bboxes))
            spike_name = images[idx].replace('.jpg', '').split('\\')[-1]
            image_folder = save_dir + spike_name + '/bbox/'
            mmcv.mkdir_or_exist(image_folder)

            draw_save = os.path.join(save_dir, spike_name, '_bboxes.png')
            bboxes_T = torch.from_numpy(bboxes).to(device='cuda', dtype=torch.float32)
            img_T = draw_bounding_boxes(img_T, bboxes_T, colors=["red"] * bboxes_T.shape[0])
            img_T = T.ToPILImage()(img_T)
            img_T.save(draw_save)

            for i, bbox in enumerate(mmcv.track_iter_progress(bboxes)):
                bbox = bbox[:-1]
                spike = img_pil.crop(bbox)
                filename = images[idx].replace('.jpg', '') + '_SPIKE_' + str(i + 1) + '.png'
                spike.save(os.path.join(image_folder, filename))
            image_folders.append(image_folder)
        elif isinstance(outputs[idx], tuple):
            (bboxes, masks) = outputs[idx]
            masks = np.squeeze(np.asarray(masks))
            bboxes = np.squeeze(np.asarray(bboxes))
            masks = masks.astype(dtype=np.uint8)
            spike_name = images[idx].replace('.jpg', '').split('\\')[-1]
            image_folder = save_dir + spike_name + '/mask/'
            mmcv.mkdir_or_exist(image_folder)

            draw_save = os.path.join(save_dir, spike_name, '_masks.png')
            masks_T = torch.from_numpy(masks).to(device='cuda', dtype=torch.bool)
            img_T = draw_segmentation_masks(img_T, masks_T, alpha=0.4, colors=["red"] * masks_T.shape[0])
            img_T = T.ToPILImage()(img_T)
            img_T.save(draw_save)

            for i, (mask, bbox) in enumerate((zip(mmcv.track_iter_progress(masks), bboxes))):
                mask = np.stack((mask,) * 3, axis=-1)
                bbox = bbox[:-1]
                result = img * mask
                result = T.ToPILImage()(result)
                result = result.crop(bbox)
                filename = spike_name + '_SPIKE_mask' + str(i + 1) + '.png'
                result.save(os.path.join(image_folder, filename))
            image_folders.append(image_folder)

    return image_folders


def count_spikelets(outputs):
    counts = []
    areas = []
    for idx in range(len(outputs)):
        mask = outputs[idx][0][:] * 255
        area = float((mask > 0.0).sum())
        # threshold
        th, threshed = cv2.threshold(mask.astype(np.ubyte), 100, 255,
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
        counts.append(len(xcnts))
        areas.append(area)
    return counts, areas


def estimate_yield(counts, areas):
    counts = np.array(counts)
    areas = np.array(areas)
    density = np.divide(counts, areas)

    sy = max(areas) * max(density) * 10
    return density, sy


def main():
    #args = parse_args()
    #img_path = args.img_dir
    img_path = 'NewFolder/'
    #output_path = args.output_dir
    output_path = 'NewFolder/'

    spike_config = 'configs/models/cascade_mask_rcnn_mod_250e_coco.py'
    spike_checkpoint = 'checkpoints/Trained/SPIKE_small/cascade_mask_rcnn_mod_epoch_250.pth'
    field_images = glob.glob(img_path + '/*.jpg', recursive=True)
    spikelet_config = 'configs/models/fcn_unet_s5-d16_32x32_80k_v3.py'
    spikelet_checkpoint = 'checkpoints/Trained/Spikelet_small_v3/fcn_unet_s5_d16_b1_iter_80k.pth'

    # Inference on field images and crop the spike results
    spike_results = inference_spike(spike_config, spike_checkpoint, field_images)
    out_folders = crop_det_bbox_mask(field_images, spike_results, output_path)

    # Inference on spike images and count spikelets
    for img, out_folder in zip(field_images, out_folders):
        spike_images = glob.glob(out_folder + '/*.png', recursive=True)
        spikelet_results = inference_spikelet(spikelet_config, spikelet_checkpoint, spike_images)
        count, area = count_spikelets(spikelet_results)

        # Estimate wheat yield based on SlypNet paper formula
        density, est = estimate_yield(count, area)

        # Save the estimation results in a text file
        img_name = img.split('\\')[-1].replace('.jpg', '')
        result_file = os.path.join(output_path, img_name, '_estimation.txt')
        with open(result_file, 'w') as f:
            f.write(f'Total Spikes: {len(spike_images)}\n'
                    f'Total Spikelets: {sum(count)}\n'
                    f'Biggest Spike: {max(area)}\n'
                    f'Most Dense Spike: {max(density):3.2e}\n'
                    f'Estimated yield: {est:.3f}')


if __name__ == '__main__':
    main()
