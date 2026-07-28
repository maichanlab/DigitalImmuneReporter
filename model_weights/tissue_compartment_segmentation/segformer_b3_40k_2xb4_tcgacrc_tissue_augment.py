checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        1024,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20250428_1024px_tiles/'
dataset_type = 'TCGACRC_TissueDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=64,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            3,
            4,
            18,
            3,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            1024,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=19,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=400, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=400,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='images', seg_map_path='tissue_masks_png'),
        data_root=
        '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20250428_1024px_tiles/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='TCGACRC_TissueDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    classwise=True, iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='images', seg_map_path='tissue_masks_png'),
        data_root=
        '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20250428_1024px_tiles/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    1024,
                    1024,
                ),
                type='RandomCrop'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                keymap=dict(gt_seg_map='mask', img='image'),
                transforms=[
                    dict(
                        min_max_height=(
                            512,
                            1024,
                        ),
                        p=0.5,
                        size=(
                            1024,
                            1024,
                        ),
                        type='RandomSizedCrop'),
                    dict(
                        p=0.5,
                        transforms=[
                            dict(
                                p=1.0,
                                per_channel=False,
                                scale=0.4,
                                type='RandomToneCurve'),
                            dict(
                                gamma_limit=(
                                    40.0,
                                    60.0,
                                ),
                                p=1.0,
                                type='RandomGamma'),
                        ],
                        type='OneOf'),
                    dict(
                        brightness=0.0,
                        contrast=0.0,
                        hue=0.1,
                        p=0.5,
                        saturation=0.5,
                        type='ColorJitter'),
                ],
                type='Albu',
                update_pad_shape=False),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='TCGACRC_TissueDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(cat_max_ratio=0.75, crop_size=(
        1024,
        1024,
    ), type='RandomCrop'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(
        keymap=dict(gt_seg_map='mask', img='image'),
        transforms=[
            dict(
                min_max_height=(
                    512,
                    1024,
                ),
                p=0.5,
                size=(
                    1024,
                    1024,
                ),
                type='RandomSizedCrop'),
            dict(
                p=0.5,
                transforms=[
                    dict(
                        p=1.0,
                        per_channel=False,
                        scale=0.4,
                        type='RandomToneCurve'),
                    dict(
                        gamma_limit=(
                            40.0,
                            60.0,
                        ), p=1.0, type='RandomGamma'),
                ],
                type='OneOf'),
            dict(
                brightness=0.0,
                contrast=0.0,
                hue=0.1,
                p=0.5,
                saturation=0.5,
                type='ColorJitter'),
        ],
        type='Albu',
        update_pad_shape=False),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='images', seg_map_path='tissue_masks_png'),
        data_root=
        '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20250428_1024px_tiles/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='TCGACRC_TissueDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    classwise=True, iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/segformer_b3_40k_2xb4_tcgacrc_tissue_augment'
