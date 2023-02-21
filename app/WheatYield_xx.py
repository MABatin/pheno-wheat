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


def inference_spike(config, checkpoint, image):
    from mmdet.apis import inference_detector, init_detector
    model = init_detector(config, checkpoint)
    results = inference_detector(model, image)

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


def crop_det_bbox_mask(image, outputs):
    spikes = []
    img = mmcv.imread(image, channel_order='rgb')
    img_pil = T.ToPILImage()(img)
    if isinstance(outputs, list):
        bboxes = outputs[0]
        bboxes = np.squeeze(np.asarray(bboxes))

        for bbox in bboxes:
            bbox = bbox[:-1]
            spike = img_pil.crop(bbox)
            spike = np.array(spike)
            spikes.append(spike)

    elif isinstance(outputs, tuple):
        (bboxes, masks) = outputs
        bboxes = bboxes[0]
        masks = np.squeeze(np.asarray(masks))
        bboxes = np.squeeze(np.asarray(bboxes))
        masks = masks.astype(dtype=np.uint8)

        for (mask, bbox) in zip(masks, bboxes):
            mask = np.stack((mask,) * 3, axis=-1)
            bbox = bbox[:-1]
            spike = img * mask
            spike = T.ToPILImage()(spike)
            spike = spike.crop(bbox)
            spike = np.array(spike)
            spikes.append(spike)

    return spikes


def count_spikelets(outputs):
    counts = []
    for idx in range(len(outputs)):
        mask = outputs[idx][0][:] * 255
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
    return counts


def measure_area(spikes):
    areas = []
    for spike in spikes:
        area = float((spike > 0.0).sum())
        areas.append(area)

    return areas


class WheatYield:
    spike_config = '../Wheat/configs/models/cascade_mask_rcnn_mod_250e_coco.py'
    spike_checkpoint = '../Wheat/checkpoints/Trained/SPIKE_small/cascade_mask_rcnn_mod_epoch_250.pth'
    spikelet_config = '../Wheat/configs/models/fcn_unet_s5-d16_32x32_80k_v3.py'
    spikelet_checkpoint = '../Wheat/checkpoints/Trained/Spikelet_small_v3/fcn_unet_s5_d16_b1_iter_80k.pth'

    def __init__(self,
                 image,
                 run=True):
        self.image = image
        self.run = run

    def inference_crop(self):
        spike_results = inference_spike(self.spike_config, self.spike_checkpoint, self.image)
        spikes = crop_det_bbox_mask(self.image, spike_results)

        return spikes, spike_results

    def draw_detection(self,
                       spike_results):
        if isinstance(spike_results, list):
            bboxes = spike_results[0]
            labels, bboxes_new = [], []
            for bbox in bboxes:
                label = bbox[-1]
                bbox = bbox[:-1]
                labels.append(f'score:{label:.2e}')
                bboxes_new.append(bbox)
            bboxes_new = np.squeeze(np.asarray(bboxes_new))
            bboxes_T = torch.from_numpy(bboxes_new).to(device='cuda', dtype=torch.float32)
            img_T = read_image(self.image)
            img_T = draw_bounding_boxes(img_T, bboxes_T, colors=["blue"] * bboxes_T.shape[0], labels=labels)
            img_T = T.ToPILImage()(img_T)

            return img_T

        elif isinstance(spike_results, tuple):
            (bboxes, masks) = spike_results
            masks = np.squeeze(np.asarray(masks))
            bboxes = bboxes[0]
            labels, bboxes_new = [], []
            for bbox in bboxes:
                label = bbox[-1]
                bbox = bbox[:-1]
                labels.append(f'score:{label:.2e}')
                bboxes_new.append(bbox)
            bboxes_new = np.squeeze(np.asarray(bboxes_new))
            masks = masks.astype(dtype=np.uint8)
            masks_T = torch.from_numpy(masks).to(device='cuda', dtype=torch.bool)
            bboxes_T = torch.from_numpy(bboxes_new).to(device='cuda', dtype=torch.float32)
            img_T = read_image(self.image)
            img_T = draw_segmentation_masks(img_T, masks_T, alpha=0.5, colors=["blue"] * masks_T.shape[0])
            img_T = draw_bounding_boxes(img_T, bboxes_T, colors=["red"] * bboxes_T.shape[0], labels=labels)
            img_T = T.ToPILImage()(img_T)

            return img_T

    def estimate_yield(self,
                       spikes):
        # Inference on spike images and count spikelets
        area = measure_area(spikes)
        spikelet_results = inference_spikelet(self.spikelet_config, self.spikelet_checkpoint, spikes)
        count = count_spikelets(spikelet_results)

        # Estimate wheat yield based on SlypNet paper formula
        count = np.array(count)
        area = np.array(area)
        density = np.divide(count, area)

        est = max(area) * max(density) * 10

        # Return the estimation results
        return density, count, area, est
