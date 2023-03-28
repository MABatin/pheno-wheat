_base_ = [
    '../../../mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '../datasets/spike_detection.py',
    '../../../mmdetection/configs/_base_/schedules/schedule_2x.py',
    '../../../mmdetection/configs/_base_/default_runtime.py'
]
device = 'cuda'
# modify num classes of the model in box head
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(
        type='BIFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        stack=3,
        norm_cfg=norm_cfg,
    ),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)))


# Set up working dir to save files and logs.
work_dir = 'Wheat/work_dirs/retinanet/'
# resume_from = 'Wheat/work_dirs/retinanet/best_bbox_mAP_epoch_12.pth'
# Change the evaluation metric since we use customized dataset.
# We can set the evaluation interval to reduce the evaluation times
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=12)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs={'entity': 'unholytsar',
                         'project': 'SpikeDetection_real',
                         'name': 'retinanet_r50_bifpn_2x_0',
                         'dir': work_dir,
                         'resume': 'allow'},
            interval=1,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=10)])
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