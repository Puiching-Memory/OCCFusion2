_base_ = ['_base_/default_runtime.py']
custom_imports = dict(imports=['occfusion'], allow_failed_imports=False)

load_from = 'ckpt/r101_dcn_fcos3d_pretrain.pth'

dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
grid_size_vt = [100, 100, 8]
num_points_per_voxel = 35
nbr_class = 17
use_lidar=True
use_radar=True
use_occ3d=False
find_unused_parameters=False

model = dict(
    type='OccFusion',
    use_occ3d=use_occ3d,
    use_lidar=use_lidar,
    use_radar=use_radar,
    data_preprocessor=dict(
        type='OccFusionDataPreprocessor',
        pad_size_divisor=32,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        voxel=True,
        voxel_layer=dict(
            grid_shape=grid_size_vt, 
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1,
        )), 
    backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),  
        stage_with_dcn=(False, False, True, True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    view_transformer=dict(
        type='MultiScaleInverseMatrixVT',
        feature_strides=[8, 16, 32],
        in_channel=[32,64,128,256],
        grid_size=[[100, 100, 8],
                   [50, 50, 4],
                   [25, 25, 2]],
        x_bound=[-50, 50],
        y_bound=[-50, 50],
        z_bound=[-5., 3.],
        sampling_rate=[4,5,6],
        num_cams=[None,None,None],
        enable_fix=False,
        use_lidar=use_lidar,
        use_radar=use_radar
        ),
    svfe_lidar=dict(
        type='SVFE',
        num_pts=num_points_per_voxel,
        input_dim=8,
        grid_size=grid_size_vt
        ),
    svfe_radar=dict(
        type='SVFE',
        num_pts=num_points_per_voxel,
        input_dim=11,
        grid_size=grid_size_vt
        ),
    occ_head=dict(
        type='OccHead',
        channels=[32,64,128,256],
        num_classes=nbr_class
        )
)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadRadarPointsMultiSweeps',
        use_occ3d=use_occ3d,
        load_dim=18,
        sweeps_num=6,
        use_dim=[0, 1, 2, 8, 9, 18],
        pc_range=point_cloud_range),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10, 
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadOccupancy'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    dict(
        type='MultiViewWrapper',
        transforms=dict(type='PhotoMetricDistortion3D')),
    dict(type='SegLabelMapping'),
    dict(
        type='Custom3DPack',
        keys=['img', 'points','pts_semantic_mask','radars','occ_200'], 
        meta_keys=['lidar2img'])
]

val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadRadarPointsMultiSweeps',
        use_occ3d=use_occ3d,
        load_dim=18,
        sweeps_num=6,
        use_dim=[0, 1, 2, 8, 9, 18],
        pc_range=point_cloud_range),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10, 
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadOccupancy'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    dict(type='SegLabelMapping'),
    dict(
        type='Custom3DPack',
        keys=['img', 'points','pts_semantic_mask','radars','occ_200'], 
        meta_keys=['lidar2img'])
]

test_pipeline = val_pipeline



train_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True), 
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_occfusion_train.pkl',
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_occfusion_val.pkl',
        pipeline=val_pipeline,
        test_mode=True)) 

test_dataloader = val_dataloader

val_evaluator = dict(type='EvalMetric')

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01), 
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=24,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24,val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))