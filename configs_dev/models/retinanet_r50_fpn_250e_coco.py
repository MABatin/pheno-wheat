_base_ = [
    '../../mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '../datasets/spike_detection.py',
    '../schedules/schedule_250e.py',
    '../../mmdetection/configs/_base_/default_runtime.py'
]
device = 'cuda'
# modify num classes of the model in box head
model = dict(
    bbox_head=dict(num_classes=1),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,  # from 0.5
            neg_iou_thr=0.5,  # from 0.4
            min_pos_iou=0.5)),  # from 0
    test_cfg=dict(
        score_thr=0.5,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# Set up working dir to save files and logs.
work_dir = 'Wheat/work_dirs/retinanet/'
resume_from = work_dir + 'latest.pth'
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs={'entity': 'unholytsar',
                         'project': 'SpikeDetection_real',
                         'name': 'retinanet_r50_fpn_250e_posneg',
                         'dir': work_dir,
                         'resume': 'allow',
                         'id': '25x35jk92b'},
            interval=1)])

auto_scale_lr = dict(enable=False, base_batch_size=16)
# Change the evaluation metric since we use customized dataset.
# We can set the evaluation interval to reduce the evaluation times
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
# Set seed thus the results are more reproducible
gpu_ids = range(1)
seed = 0
workflow = [('train', 1), ('val', 1)]

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
