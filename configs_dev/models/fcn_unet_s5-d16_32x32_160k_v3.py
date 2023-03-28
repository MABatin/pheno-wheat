_base_ = [
    '../datasets/spikelet_segmentation.py',
    '../../../mmsegmentation/configs/_base_/default_runtime.py',
    '../schedules/schedule_160k.py'
]
data_root = 'Wheat/data/Spikelet_small_v1'
# model settings
device = 'cuda'
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='GN', num_groups=4, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=16,  # 16 for b1
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,  # 16 for b1
        in_index=4,
        channels=16,  # 16 for b1
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.51, 24.4])),  # added mfb class_weight
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=32,  # 32 for b1
        in_index=3,
        channels=16,  # 16 for b1
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.51, 24.4])),  # added mfb class_weight
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(32, 32), stride=(21, 21)))

# Set up working dir to save files and logs.
work_dir = 'Wheat/work_dirs/fcn_unet_s5_d16/'
# resume_from = work_dir + 'iter_6000.pth'
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            by_epoch=False,
            init_kwargs={'entity': 'unholytsar',
                         'project': 'SpikeletSegm',
                         'name': 'fcn_unet_s5_d16_160k_v3_c1',
                         'dir': work_dir,
                         'resume': 'allow',
                         'id': '2ksi3syaidrtyr'},
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
