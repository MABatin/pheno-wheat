# optimizer
optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[100, 140])
runner = dict(type='EpochBasedRunner', max_epochs=150)  # Reduce the learning rate in 100 and 140 epoch

