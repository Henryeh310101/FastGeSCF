"""
Change Detection Model Module:

This module contains models that solve image change detection problem, where
image change detection is find the difference between a image pair.

"""
import time
import torch
import torch.nn as nn

from .transformer import (
    TwoCrossAttention,
    CrossAttentionBlock,
)
from model.pseudo_generator import PseudoGenerator

class Encoder(nn.Module):
    def __init__(
        self,
        num_heads=1,
        dim=576,
        img_size=(512,512),
        device="auto",
        sam2_checkpoint=None,
    ):
        ### model dim ###
        # Large:     576
        # Base_plus: 448
        # Small:     384
        ##################################
        super(Encoder, self).__init__()
        self.qkv_generator = PseudoGenerator(
            feature_layer=24,
            img_size=img_size,
            device=device,
            sam2_checkpoint=sam2_checkpoint,
        )
        self.num_heads = num_heads
        self.dim = dim
        self.freeze_module(self.qkv_generator)

    def freeze_module(self, module):
        """Helper function to freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False
    
    def forward(self, img_1, img_2):
        t0_key, t1_key, t0_embed, t1_embed = self.qkv_generator(img_1, img_2)
        return t0_key, t1_key, t0_embed, t1_embed


class CrossAttention(Encoder):
    def __init__(
        self,
        num_heads=1,
        dropout_rate=0.1,
        target_shp=(512, 512),
        num_blocks=2,
        dim=576,
        device="auto",
        sam2_checkpoint=None,
        **kwargs,
    ):
        # initialize TwoDinoSingleUnet (backbone)
        super().__init__(img_size=target_shp, device=device, sam2_checkpoint=sam2_checkpoint)

        ##### interactor #####
        ca0s = []

        for _ in range(num_blocks):

            ca0s.append(
                TwoCrossAttention(
                    self.dim,
                    num_heads,
                    dropout_rate,
                    self.dim,
                    num_heads,
                    dropout_rate,
                )
            )

        self.ca0s = nn.ModuleList(ca0s)

        ##### decoder ######
        self.conv1 = nn.Conv2d(
            self.dim * 2,
            self.dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            self.dim, 2, kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(
            size=target_shp, mode="bilinear", align_corners=False
        )

    
    @staticmethod
    def _reshape_before_CA(x):
        batch, m, n, feat = x.shape
        x = x.reshape(batch, m * n, feat)
        return (batch, feat, m, n), x

    @staticmethod
    def _reshape_after_CA(x, origin_shp):
        x = x.permute(0, 2, 1)  # (batch, feat, m * n)
        return x.reshape(*origin_shp)  # (batch, feat, m, n)

    @staticmethod
    def apply_cross_attention(self, x_origin, y_origin, cas):
        shp_x, x = self._reshape_before_CA(x_origin)
        shp_y, y = self._reshape_before_CA(y_origin)
        
        for ca in cas:
            x, y = ca(x, y, y, y, x, x)

        x = self._reshape_after_CA(x, shp_x)
        y = self._reshape_after_CA(y, shp_y)
        return x, y

    def forward(self, img_1, img_2):        
        ### origianl order ###
        feat_1_0, feat_2_0, _, _ = super().forward(img_1, img_2)
        
        # cross attention on (row, col)
        x_1_0, x_2_0 = self.apply_cross_attention(
            self, feat_1_0, feat_2_0, self.ca0s
        )

        # (batch, 2f, r, c)
        x = torch.concatenate([x_1_0, x_2_0], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        ### Before Upsample             ###
        ### torch.Size([1, 2, 32, 32])  ###
        x_upsample = self.upsample(x)
        x_upsample = x_upsample.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        
        # ### reverse order ###
        feat_1_0, feat_2_0, _,_ = super().forward(img_2, img_1)

        # cross attention on (row, col)
        x_1_0, x_2_0 = self.apply_cross_attention(
            self, feat_1_0, feat_2_0, self.ca0s
        )

        # (batch, 2f, r, c)
        x_reverse = torch.concatenate([x_1_0, x_2_0], dim=1)
        x_reverse = self.relu(self.conv1(x_reverse))
        x_reverse = self.relu(self.conv2(x_reverse))
        x_reverse_upsample = self.upsample(x_reverse)
        x_reverse_upsample = x_reverse_upsample.permute(0, 2, 3, 1)  # (batch, *target_shp, 2)
        
        return x_upsample, x_reverse_upsample, x, x_reverse
