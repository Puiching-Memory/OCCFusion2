import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.runner.amp import autocast


@MODELS.register_module()
class OccHead2(BaseModule):
    def __init__(self, channels, num_classes):
        super(OccHead2, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(channels, 256, 1),
            nn.SiLU(),
            nn.Conv2d(256, num_classes, 1),
        )

    @autocast("cuda", torch.float32)
    def forward(self, x):

        print(len(x),x[0].shape)
        

        x = self.head(x)

        return x
        # torch.Size([1, 200, 200, 16, 17]) B, x, y, z, cls
