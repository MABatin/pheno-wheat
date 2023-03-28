_base_ = [
    '../../../mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../datasets/spike_detection.py',
    '../../../mmdetection/configs/_base_/schedules/schedule_2x.py',
    '../../../mmdetection/configs/_base_/default_runtime.py'
]

device = 'cuda'
# modify num classes of the model in box head
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.3,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

# load_from = 'Wheat/checkpoints/Downloaded/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
work_dir = 'Wheat/work_dirs/faster_rcnn/'

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs={'entity': 'unholytsar',
                         'project': 'SpikeDetection_real',
                         'name': 'faster_rcnn_r50_fpn_2x_2',
                         'dir': work_dir},
            interval=1)])

# Set up working dir to save files and logs.

# Change the evaluation metric since we use customized dataset.
# We can set the evaluation interval to reduce the evaluation times
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=12)
# Set seed thus the results are more reproducible
gpu_ids = range(1)
seed = 0
workflow = [('train', 1), ('val', 1)]  # val epoch doesn't update parameters
# Extra settings
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[16, 22])  # Reduce the learning rate in 16 and 22 epoch
auto_scale_lr = dict(enable=False, base_batch_size=16)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# should do - lr=0.01, warmup_ratio=.01
