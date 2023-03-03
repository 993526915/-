rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(3, ),
        drop_path_rate=0.1,
        gap_before_final_norm=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128-noema_in1k_20220222-2908964a.pth',
            prefix='backbone')),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            mode='original')))
size = (256, 256)
new_prefix = '/HARD-DATA/qch/mmclassification/data/Server/'
train_data_prefix = '/HARD-DATA/qch/mmclassification/data/Server/train'
val_data_prefix = '/HARD-DATA/qch/mmclassification/data/Server/val'
test_data_prefix = '/HARD-DATA/qch/mmclassification/data/Server/test'
train_ann_file = '/HARD-DATA/qch/mmclassification/data/Server/meta/train.txt'
val_ann_file = '/HARD-DATA/qch/mmclassification/data/Server/meta/val.txt'
test_ann_file = '/HARD-DATA/qch/mmclassification/data/Server/meta/test.txt'
train_mean = [0.79036833, 0.61805679, 0.79187009]
train_std = [0.107446, 0.18938092, 0.13786516]
val_mean = [0.79002808, 0.61708047, 0.79179805]
val_std = [0.10753236, 0.18973679, 0.13806012]
test_mean = [0.78786622, 0.60472527, 0.79599074]
test_std = [0.09341277, 0.17579078, 0.12526942]
classes = ['t0', 't1', 't2', 't3', 'tis']
samples_per_gpu = 72
workers_per_gpu = 60
dataset_type = 'CustomDataset'
train_img_norm_cfg = dict(
    mean=[0.79036833, 0.61805679, 0.79187009],
    std=[0.107446, 0.18938092, 0.13786516],
    to_rgb=True)
val_img_norm_cfg = dict(
    mean=[0.79002808, 0.61708047, 0.79179805],
    std=[0.10753236, 0.18973679, 0.13806012],
    to_rgb=True)
test_img_norm_cfg = dict(
    mean=[0.78786622, 0.60472527, 0.79599074],
    std=[0.09341277, 0.17579078, 0.12526942],
    to_rgb=True)
evaluation = dict(
    interval=1,
    start=1,
    save_best='auto',
    metric='accuracy',
    metric_options=dict(topk=(1, )))
transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.01,
        scale_limit=0.1,
        rotate_limit=15,
        interpolation=1,
        p=0.5),
    dict(type='CLAHE', clip_limit=4.0, p=0.5),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256)),
    dict(
        type='RandAugment',
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Invert'),
            dict(
                type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
            dict(
                type='Posterize', magnitude_key='bits',
                magnitude_range=(4, 0)),
            dict(
                type='Solarize', magnitude_key='thr',
                magnitude_range=(256, 0)),
            dict(
                type='SolarizeAdd',
                magnitude_key='magnitude',
                magnitude_range=(0, 110)),
            dict(
                type='ColorTransform',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Contrast',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Brightness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Sharpness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='horizontal'),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='vertical'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='horizontal'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='vertical')
        ],
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[1, 1, 1], interpolation='bicubic')),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.01,
                scale_limit=0.1,
                rotate_limit=15,
                interpolation=1,
                p=0.5),
            dict(type='CLAHE', clip_limit=4.0, p=0.5),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='VerticalFlip', p=0.5)
        ]),
    dict(
        type='Normalize',
        mean=[0.79036833, 0.61805679, 0.79187009],
        std=[0.107446, 0.18938092, 0.13786516],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256)),
    dict(
        type='RandAugment',
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Invert'),
            dict(
                type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
            dict(
                type='Posterize', magnitude_key='bits',
                magnitude_range=(4, 0)),
            dict(
                type='Solarize', magnitude_key='thr',
                magnitude_range=(256, 0)),
            dict(
                type='SolarizeAdd',
                magnitude_key='magnitude',
                magnitude_range=(0, 110)),
            dict(
                type='ColorTransform',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Contrast',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Brightness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Sharpness',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.9)),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='horizontal'),
            dict(
                type='Shear',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.3),
                direction='vertical'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='horizontal'),
            dict(
                type='Translate',
                magnitude_key='magnitude',
                magnitude_range=(0, 0.45),
                direction='vertical')
        ],
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[1, 1, 1], interpolation='bicubic')),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.01,
                scale_limit=0.1,
                rotate_limit=15,
                interpolation=1,
                p=0.5),
            dict(type='CLAHE', clip_limit=4.0, p=0.5),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='VerticalFlip', p=0.5)
        ]),
    dict(
        type='Normalize',
        mean=[0.79002808, 0.61708047, 0.79179805],
        std=[0.10753236, 0.18973679, 0.13806012],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256), backend='pillow'),
    dict(
        type='Normalize',
        mean=[0.78786622, 0.60472527, 0.79599074],
        std=[0.09341277, 0.17579078, 0.12526942],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=72,
    workers_per_gpu=60,
    train=dict(
        type='CustomDataset',
        data_prefix='/HARD-DATA/qch/mmclassification/data/Server/train',
        ann_file='/HARD-DATA/qch/mmclassification/data/Server/meta/train.txt',
        classes=['t0', 't1', 't2', 't3', 'tis'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(
                type='RandAugment',
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(type='Invert'),
                    dict(
                        type='Rotate',
                        magnitude_key='angle',
                        magnitude_range=(0, 30)),
                    dict(
                        type='Posterize',
                        magnitude_key='bits',
                        magnitude_range=(4, 0)),
                    dict(
                        type='Solarize',
                        magnitude_key='thr',
                        magnitude_range=(256, 0)),
                    dict(
                        type='SolarizeAdd',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 110)),
                    dict(
                        type='ColorTransform',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Contrast',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Brightness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Sharpness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='horizontal'),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='vertical'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='horizontal'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='vertical')
                ],
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[1, 1, 1], interpolation='bicubic')),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.01,
                        scale_limit=0.1,
                        rotate_limit=15,
                        interpolation=1,
                        p=0.5),
                    dict(type='CLAHE', clip_limit=4.0, p=0.5),
                    dict(type='HorizontalFlip', p=0.5),
                    dict(type='VerticalFlip', p=0.5)
                ]),
            dict(
                type='Normalize',
                mean=[0.79036833, 0.61805679, 0.79187009],
                std=[0.107446, 0.18938092, 0.13786516],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        data_prefix='/HARD-DATA/qch/mmclassification/data/Server/val',
        ann_file='/HARD-DATA/qch/mmclassification/data/Server/meta/val.txt',
        classes=['t0', 't1', 't2', 't3', 'tis'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(
                type='RandAugment',
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(type='Invert'),
                    dict(
                        type='Rotate',
                        magnitude_key='angle',
                        magnitude_range=(0, 30)),
                    dict(
                        type='Posterize',
                        magnitude_key='bits',
                        magnitude_range=(4, 0)),
                    dict(
                        type='Solarize',
                        magnitude_key='thr',
                        magnitude_range=(256, 0)),
                    dict(
                        type='SolarizeAdd',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 110)),
                    dict(
                        type='ColorTransform',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Contrast',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Brightness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Sharpness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='horizontal'),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='vertical'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='horizontal'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='vertical')
                ],
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[1, 1, 1], interpolation='bicubic')),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.01,
                        scale_limit=0.1,
                        rotate_limit=15,
                        interpolation=1,
                        p=0.5),
                    dict(type='CLAHE', clip_limit=4.0, p=0.5),
                    dict(type='HorizontalFlip', p=0.5),
                    dict(type='VerticalFlip', p=0.5)
                ]),
            dict(
                type='Normalize',
                mean=[0.79002808, 0.61708047, 0.79179805],
                std=[0.10753236, 0.18973679, 0.13806012],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='CustomDataset',
        data_prefix='/HARD-DATA/qch/mmclassification/data/Server/test',
        ann_file='/HARD-DATA/qch/mmclassification/data/Server/meta/test.txt',
        classes=['t0', 't1', 't2', 't3', 'tis'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256), backend='pillow'),
            dict(
                type='Normalize',
                mean=[0.78786622, 0.60472527, 0.79599074],
                std=[0.09341277, 0.17579078, 0.12526942],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
lr = 0.0003
weight_decay = 0.05
momentum = 0.9
max_norm = 5.0
warmup_by_epoch = True
warmup_iters = 2
min_lr_ratio = 0.01
max_epochs = 25
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys=dict({
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW', lr=0.0003, weight_decay=0.05, eps=0.0001, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_iters=2,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=25)
checkpoint_config = dict(
    interval=1,
    by_epoch=True,
    save_optimizer=False,
    max_keep_ckpts=-1,
    save_last=False)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '../work_dirs/model-60G'
gpu_ids = range(0, 3)
