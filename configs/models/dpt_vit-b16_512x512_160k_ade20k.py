_base_ = [
    '../datasets/spikelet_segmentation.py',
    '../../../mmsegmentation/configs/_base_/default_runtime.py',
    '../schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='../../../mmsegmentation/pretrain/vit-b16_p16_224-80ecf9dd.pth', # noqa
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        out_indices=(2, 5, 8, 11),
        final_norm=False,
        with_cls_token=True,
        output_cls_token=True),
    decode_head=dict(
        type='DPTHead',
        in_channels=(768, 768, 768, 768),
        channels=256,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        num_classes=150,
        readout_type='project',
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(32, 32), stride=(21, 21)))  # yapf: disable

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2, workers_per_gpu=0)
# Set up working dir to save files and logs.
work_dir = 'Wheat/work_dirs/fcn_unet_s5_d16/'
# resume_from = work_dir + 'epoch_18.pth'
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            by_epoch=False,
            init_kwargs={'entity': 'unholytsar',
                         'project': 'SpikeletSegm',
                         'name': 'fcn_unet_s5_d16_40k',
                         'dir': work_dir,
                         'resume': 'allow',
                         'id': '2ksi3syaix'},
            interval=1)])

auto_scale_lr = dict(enable=False, base_batch_size=16)
# Change the evaluation metric since we use customized dataset.
# We can set the evaluation interval to reduce the evaluation times
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=1000, max_keep_ckpts=3, by_epoch=False)
# Set seed thus the results are more reproducible
gpu_ids = range(1)
seed = 0
workflow = [('train', 1)]
evaluation = dict(metric=['mDice', 'mIoU'], interval=1000)
