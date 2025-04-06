from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from focal_loss.focal_loss import FocalLoss
from mmdet3d.models.losses import LovaszLoss
from .loss import geo_scal_loss, sem_scal_loss
from typing import Dict
from mmdet3d.structures.det3d_data_sample import (
    ForwardResults,
    OptSampleList,
    SampleList,
)


@MODELS.register_module()
class OccFusion2(Base3DSegmentor):
    def __init__(
        self,
        data_preprocessor,
        backbone_cfg,
        neck_cfg,
        view_transformer_cfg,
        occ_head_cfg,
    ):
        super().__init__(data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone_cfg)
        self.neck = MODELS.build(neck_cfg)
        self.view_transformer = MODELS.build(view_transformer_cfg)
        self.occ_head = MODELS.build(occ_head_cfg)
        #self.loss_fl = FocalLoss(gamma=2, ignore_index=255)
        self.loss_cls = nn.CrossEntropyLoss()

    def extract_feat(self, batch_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features of images."""
        # N = 输入视图数量,例如:输入6视图
        B, N, C, H, W = batch_inputs.size()
        batch_inputs = batch_inputs.reshape(B * N, C, H, W)

        x = self.backbone(batch_inputs)
        x = self.neck(x)

        return {
            "imgs": x, # tuple[Tensor,...]
        }

    def _forward(self, batch_inputs: dict, batch_data_samples: OptSampleList = None) -> torch.Tensor:
        """Forward training function."""
        # batch_inputs: dict_keys(['points', 'voxels', 'lidar_voxel_feats', 'lidar_voxel_coords', 'radar_voxel_feats', 'radar_voxel_coords', 'dense_occ_200', 'imgs'])
        # imgs: torch.Size([1, 6, 3, 928, 1600])

        # batch_data_samples: list[Det3DDataSample]
        # lidar2img,pad_shape,batch_input_shape,point_coors,gt_instances_3d,gt_instances,gt_pts_seg,eval_ann_info

        img_feats = self.extract_feat(batch_inputs["imgs"])

        outputs = img_feats["imgs"]

        return outputs

    def loss(self, batch_inputs:dict, batch_data_samples:SampleList) -> Dict[str, torch.Tensor]:
        batch_outputs = self._forward(batch_inputs, batch_data_samples)

        loss1 = self.loss_cls(batch_outputs[0],batch_outputs[0])

        return {"loss":loss1}

    def predict(self, batch_inputs, batch_data_samples):
        """Forward predict function."""
        outputs = self._forward(batch_inputs, batch_data_samples)

        outputs = self.postprocess_result(outputs, batch_data_samples)

        return outputs

    def encode_decode(self, batch_inputs, batch_data_samples):
        return
