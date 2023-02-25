_base_ = [
    '../datasets/spike_instance.py',
    '../../mmdetection/configs/_base_/default_runtime.py',
    '../schedules/schedule_150e.py'
]
device = 'cuda'
# modify num classes of the model in box head
norm_cfg = dict(type='BN', requires_grad=True)
# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5)],
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


# Set up working dir to save files and logs.
work_dir = 'Wheat/work_dirs/retinanet/'
# resume_from = work_dir+'epoch_201.pth'  # change to work_dir+'latest.pth' if training is interrupted
TAGS = ['150 epochs']
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs={'dir': work_dir,
                         'resume': 'allow',
                         'tags': TAGS},
            interval=1,
            num_eval_images=0,
            bbox_score_thr=0.5,
            log_checkpoint=False,
            log_checkpoint_metadata=False)])

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
