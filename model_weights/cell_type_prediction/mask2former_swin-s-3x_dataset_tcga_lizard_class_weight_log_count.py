auto_scale_lr = dict(base_batch_size=16, enable=False)
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
backend_args = None
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            256,
            256,
        ),
        type='BatchFixedSizePad'),
]
classes = (
    'Background',
    'Neutrophil',
    'Tumor',
    'Lymphocyte',
    'Plasmacell',
    'Eosinophil',
    'Other',
)
combined_train_dataset = dict(
    datasets=[
        dict(
            ann_file=
            '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/train/coco_annotations.json',
            backend_args=None,
            data_prefix=dict(
                img=
                '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/train/images/'
            ),
            filter_cfg=dict(filter_empty_gt=True),
            metainfo=dict(
                classes=(
                    'Background',
                    'Neutrophil',
                    'Tumor',
                    'Lymphocyte',
                    'Plasmacell',
                    'Eosinophil',
                    'Other',
                )),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(
                    poly2mask=True,
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True),
                dict(keep_ratio=True, scale=(
                    256,
                    256,
                ), type='Resize'),
                dict(
                    bbox_params=dict(
                        filter_lost_elements=True,
                        format='pascal_voc',
                        label_fields=[
                            'gt_bboxes_labels',
                            'gt_ignore_flags',
                        ],
                        min_visibility=0.0,
                        type='BboxParams'),
                    keymap=dict(
                        gt_bboxes='bboxes', gt_masks='masks', img='image'),
                    skip_img_without_anno=True,
                    transforms=[
                        dict(
                            p=0.5,
                            transforms=[
                                dict(p=1.0, type='ChannelShuffle'),
                                dict(
                                    b_shift_limit=20,
                                    g_shift_limit=0,
                                    p=1.0,
                                    r_shift_limit=20,
                                    type='RGBShift'),
                            ],
                            type='OneOf'),
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
                    type='Albu'),
                dict(
                    allow_negative_crop=False,
                    crop_size=(
                        0.5,
                        0.5,
                    ),
                    crop_type='relative_range',
                    recompute_bbox=True,
                    type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        dict(
            ann_file=
            '/data_g1/AI_projects/processed_data/Lizard_CRC/20260112_256px_0.5mpp_tiles/train/coco_annotations.json',
            backend_args=None,
            data_prefix=dict(
                img=
                '/data_g1/AI_projects/processed_data/Lizard_CRC/20260112_256px_0.5mpp_tiles/train/images/'
            ),
            filter_cfg=dict(filter_empty_gt=True),
            metainfo=dict(
                classes=(
                    'Background',
                    'Neutrophil',
                    'Tumor',
                    'Lymphocyte',
                    'Plasmacell',
                    'Eosinophil',
                    'Other',
                )),
            pipeline=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(
                    poly2mask=True,
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True),
                dict(keep_ratio=True, scale=(
                    256,
                    256,
                ), type='Resize'),
                dict(
                    bbox_params=dict(
                        filter_lost_elements=True,
                        format='pascal_voc',
                        label_fields=[
                            'gt_bboxes_labels',
                            'gt_ignore_flags',
                        ],
                        min_visibility=0.0,
                        type='BboxParams'),
                    keymap=dict(
                        gt_bboxes='bboxes', gt_masks='masks', img='image'),
                    skip_img_without_anno=True,
                    transforms=[
                        dict(
                            p=0.5,
                            transforms=[
                                dict(p=1.0, type='ChannelShuffle'),
                                dict(
                                    b_shift_limit=20,
                                    g_shift_limit=0,
                                    p=1.0,
                                    r_shift_limit=20,
                                    type='RGBShift'),
                            ],
                            type='OneOf'),
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
                    type='Albu'),
                dict(
                    allow_negative_crop=False,
                    crop_size=(
                        0.5,
                        0.5,
                    ),
                    crop_type='relative_range',
                    recompute_bbox=True,
                    type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
    ],
    type='ConcatDataset')
custom_keys = dict({
    'absolute_pos_embed':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.patch_embed.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.10.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.11.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.12.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.13.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.14.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.15.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.16.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.17.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.3.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.4.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.5.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.6.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.7.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.8.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.9.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'level_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat':
    dict(decay_mult=0.0, lr_mult=1.0),
    'relative_position_bias_table':
    dict(decay_mult=0.0, lr_mult=0.1)
})
data_preprocessor = dict(
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=False,
            size=(
                256,
                256,
            ),
            type='BatchFixedSizePad'),
    ],
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_seg=True,
    pad_size_divisor=32,
    seg_pad_value=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
depths = [
    2,
    2,
    18,
    2,
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    256,
    256,
)
launcher = 'none'
lizard_dataset = dict(
    ann_file=
    '/data_g1/AI_projects/processed_data/Lizard_CRC/20260112_256px_0.5mpp_tiles/train/coco_annotations.json',
    backend_args=None,
    data_prefix=dict(
        img=
        '/data_g1/AI_projects/processed_data/Lizard_CRC/20260112_256px_0.5mpp_tiles/train/images/'
    ),
    filter_cfg=dict(filter_empty_gt=True),
    metainfo=dict(
        classes=(
            'Background',
            'Neutrophil',
            'Tumor',
            'Lymphocyte',
            'Plasmacell',
            'Eosinophil',
            'Other',
        )),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(
            poly2mask=True,
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True),
        dict(keep_ratio=True, scale=(
            256,
            256,
        ), type='Resize'),
        dict(
            bbox_params=dict(
                filter_lost_elements=True,
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                min_visibility=0.0,
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
            skip_img_without_anno=True,
            transforms=[
                dict(
                    p=0.5,
                    transforms=[
                        dict(p=1.0, type='ChannelShuffle'),
                        dict(
                            b_shift_limit=20,
                            g_shift_limit=0,
                            p=1.0,
                            r_shift_limit=20,
                            type='RGBShift'),
                    ],
                    type='OneOf'),
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
            type='Albu'),
        dict(
            allow_negative_crop=False,
            crop_size=(
                0.5,
                0.5,
            ),
            crop_type='relative_range',
            recompute_bbox=True,
            type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PackDetInputs'),
    ],
    type='CocoDataset')
load_from = 'work_dirs/mask2former_swin-s-3x_dataset_tcga_lizard_class_weight_log_count/epoch_36.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
logger = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    interval=50)
max_epochs = 12
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=96,
        frozen_stages=-1,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=False),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                img_pad_value=0,
                mask_pad_value=0,
                pad_mask=True,
                pad_seg=False,
                size=(
                    256,
                    256,
                ),
                type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_seg=True,
        pad_size_divisor=32,
        seg_pad_value=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes=7,
        type='MaskFormerFusionHead'),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        loss_cls=dict(
            class_weight=[
                0,
                1.29,
                0.9,
                1.08,
                1.36,
                1.44,
                0.93,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_queries=100,
        num_stuff_classes=0,
        num_things_classes=7,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=300,
        panoptic_on=False,
        semantic_on=False),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler')),
    type='Mask2Former')
num_classes = 7
num_stuff_classes = 0
num_things_classes = 7
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'absolute_pos_embed':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone':
            dict(decay_mult=1.0, lr_mult=0.1),
            'backbone.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.patch_embed.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.10.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.11.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.12.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.13.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.14.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.15.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.16.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.17.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.2.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.3.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.4.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.5.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.6.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.7.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.8.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.9.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'level_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat':
            dict(decay_mult=0.0, lr_mult=1.0),
            'relative_position_bias_table':
            dict(decay_mult=0.0, lr_mult=0.1)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            28,
            32,
        ],
        type='MultiStepLR'),
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
resume = False
tcgacrc_dataset = dict(
    ann_file=
    '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/train/coco_annotations.json',
    backend_args=None,
    data_prefix=dict(
        img=
        '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/train/images/'
    ),
    filter_cfg=dict(filter_empty_gt=True),
    metainfo=dict(
        classes=(
            'Background',
            'Neutrophil',
            'Tumor',
            'Lymphocyte',
            'Plasmacell',
            'Eosinophil',
            'Other',
        )),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(
            poly2mask=True,
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True),
        dict(keep_ratio=True, scale=(
            256,
            256,
        ), type='Resize'),
        dict(
            bbox_params=dict(
                filter_lost_elements=True,
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                min_visibility=0.0,
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
            skip_img_without_anno=True,
            transforms=[
                dict(
                    p=0.5,
                    transforms=[
                        dict(p=1.0, type='ChannelShuffle'),
                        dict(
                            b_shift_limit=20,
                            g_shift_limit=0,
                            p=1.0,
                            r_shift_limit=20,
                            type='RGBShift'),
                    ],
                    type='OneOf'),
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
            type='Albu'),
        dict(
            allow_negative_crop=False,
            crop_size=(
                0.5,
                0.5,
            ),
            crop_type='relative_range',
            recompute_bbox=True,
            type='RandomCrop'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PackDetInputs'),
    ],
    type='CocoDataset')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file=
        '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/val/coco_annotations.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/val/images/'
        ),
        metainfo=dict(
            classes=(
                'Background',
                'Neutrophil',
                'Tumor',
                'Lymphocyte',
                'Plasmacell',
                'Eosinophil',
                'Other',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/val/coco_annotations.json',
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=36)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file=
                '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/train/coco_annotations.json',
                backend_args=None,
                data_prefix=dict(
                    img=
                    '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/train/images/'
                ),
                filter_cfg=dict(filter_empty_gt=True),
                metainfo=dict(
                    classes=(
                        'Background',
                        'Neutrophil',
                        'Tumor',
                        'Lymphocyte',
                        'Plasmacell',
                        'Eosinophil',
                        'Other',
                    )),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(
                        poly2mask=True,
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=True),
                    dict(keep_ratio=True, scale=(
                        256,
                        256,
                    ), type='Resize'),
                    dict(
                        bbox_params=dict(
                            filter_lost_elements=True,
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            min_visibility=0.0,
                            type='BboxParams'),
                        keymap=dict(
                            gt_bboxes='bboxes', gt_masks='masks', img='image'),
                        skip_img_without_anno=True,
                        transforms=[
                            dict(
                                p=0.5,
                                transforms=[
                                    dict(p=1.0, type='ChannelShuffle'),
                                    dict(
                                        b_shift_limit=20,
                                        g_shift_limit=0,
                                        p=1.0,
                                        r_shift_limit=20,
                                        type='RGBShift'),
                                ],
                                type='OneOf'),
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
                        type='Albu'),
                    dict(
                        allow_negative_crop=False,
                        crop_size=(
                            0.5,
                            0.5,
                        ),
                        crop_type='relative_range',
                        recompute_bbox=True,
                        type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PackDetInputs'),
                ],
                type='CocoDataset'),
            dict(
                ann_file=
                '/data_g1/AI_projects/processed_data/Lizard_CRC/20260112_256px_0.5mpp_tiles/train/coco_annotations.json',
                backend_args=None,
                data_prefix=dict(
                    img=
                    '/data_g1/AI_projects/processed_data/Lizard_CRC/20260112_256px_0.5mpp_tiles/train/images/'
                ),
                filter_cfg=dict(filter_empty_gt=True),
                metainfo=dict(
                    classes=(
                        'Background',
                        'Neutrophil',
                        'Tumor',
                        'Lymphocyte',
                        'Plasmacell',
                        'Eosinophil',
                        'Other',
                    )),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(
                        poly2mask=True,
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=True),
                    dict(keep_ratio=True, scale=(
                        256,
                        256,
                    ), type='Resize'),
                    dict(
                        bbox_params=dict(
                            filter_lost_elements=True,
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            min_visibility=0.0,
                            type='BboxParams'),
                        keymap=dict(
                            gt_bboxes='bboxes', gt_masks='masks', img='image'),
                        skip_img_without_anno=True,
                        transforms=[
                            dict(
                                p=0.5,
                                transforms=[
                                    dict(p=1.0, type='ChannelShuffle'),
                                    dict(
                                        b_shift_limit=20,
                                        g_shift_limit=0,
                                        p=1.0,
                                        r_shift_limit=20,
                                        type='RGBShift'),
                                ],
                                type='OneOf'),
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
                        type='Albu'),
                    dict(
                        allow_negative_crop=False,
                        crop_size=(
                            0.5,
                            0.5,
                        ),
                        crop_type='relative_range',
                        recompute_bbox=True,
                        type='RandomCrop'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PackDetInputs'),
                ],
                type='CocoDataset'),
        ],
        type='ConcatDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='Resize'),
    dict(
        bbox_params=dict(
            filter_lost_elements=True,
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            min_visibility=0.0,
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
        skip_img_without_anno=True,
        transforms=[
            dict(
                p=0.5,
                transforms=[
                    dict(p=1.0, type='ChannelShuffle'),
                    dict(
                        b_shift_limit=20,
                        g_shift_limit=0,
                        p=1.0,
                        r_shift_limit=20,
                        type='RGBShift'),
                ],
                type='OneOf'),
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
        type='Albu'),
    dict(
        allow_negative_crop=False,
        crop_size=(
            0.5,
            0.5,
        ),
        crop_type='relative_range',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file=
        '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/val/coco_annotations.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/val/images/'
        ),
        metainfo=dict(
            classes=(
                'Background',
                'Neutrophil',
                'Tumor',
                'Lymphocyte',
                'Plasmacell',
                'Eosinophil',
                'Other',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/data_g1/AI_projects/processed_data/TCGA_CRC/TCGA_CRC_Datasets/20260105_256px_0.5mpp_tiles/val/coco_annotations.json',
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/mask2former_swin-s-3x_dataset_tcga_lizard_class_weight_log_count'
