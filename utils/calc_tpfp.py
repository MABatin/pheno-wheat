import json
import os.path
import traceback

from mmdet.core.evaluation.mean_ap import tpfp_default


def tpfp2json(cfg, dataset, outputs, tpfp_prefix, iou_thr):
    json_dict = dict()
    gts = 0
    dets = 0
    total_tp = 0
    total_fp = 0
    data_info = dataset.load_annotations(cfg.data.test.ann_file)
    json_dict['images'] = []
    json_dict['metrics'] = []
    for idx in range(len(outputs)):
        img_dict = dict()
        image = data_info[idx]['file_name']
        gt_bboxes = dataset.get_ann_info(idx)['bboxes']
        gt_bboxes_ignore = dataset.get_ann_info(idx)['bboxes_ignore']
        try:
            if isinstance(outputs[idx], list):
                (tp, fp) = tpfp_default(outputs[idx][0], gt_bboxes, gt_bboxes_ignore, iou_thr=iou_thr)
                img_tp = 0
                img_fp = 0
                for i in tp[0]:
                    if i == 1.0:
                        total_tp += 1
                        img_tp += 1
                for i in fp[0]:
                    if i == 1.0:
                        total_fp += 1
                        img_fp += 1
                gt = gt_bboxes.shape[0]
                det = outputs[idx][0].shape[0]
                gts += gt
                dets += det
            elif isinstance(outputs[idx], tuple):
                (tp, fp) = tpfp_default(outputs[idx][0][0], gt_bboxes, gt_bboxes_ignore, iou_thr=iou_thr)
                img_tp = 0
                img_fp = 0
                for i in tp[0]:
                    if i == 1.0:
                        total_tp += 1
                        img_tp += 1
                for i in fp[0]:
                    if i == 1.0:
                        total_fp += 1
                        img_fp += 1
                gt = gt_bboxes.shape[0]
                det = outputs[idx][0][0].shape[0]
                gts += gt
                dets += det

            img_dict['img_id'] = idx
            img_dict['img_name'] = os.path.basename(image)
            img_dict['ground_truth'] = gt
            img_dict['detected'] = det
            img_dict['true_positive'] = img_tp
            img_dict['false_positive'] = img_fp
            img_dict['false_negative'] = gt_bboxes.shape[0] - img_tp
            json_dict['images'].append(img_dict)
        except Exception as e:
            print(e)
            print(f'outputs must be either list or tuple not {type(outputs[idx])}')

    total_dict = {
        "total_gt": gts,
        "total_det": dets,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": gts - total_tp
    }
    json_dict['metrics'].append(total_dict)
    precision = total_tp / dets
    recall = total_tp / gts
    accuracy = total_tp / (gts + total_fp)
    f1_score = (2 * precision * recall) / (precision + recall)
    metric_dict = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4)
    }
    json_dict['metrics'].append(metric_dict)

    json_obj = json.dumps(json_dict, indent=4)
    with open(tpfp_prefix, 'w') as f:
        f.write(json_obj)
        f.close()
