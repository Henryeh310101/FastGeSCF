import torch
import torch.nn as nn

from .transformer import (
    TwoCrossAttention,
)
from model.pseudo_generator import PseudoGenerator


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads=1,
        dim=576,
        img_size=(512,512), 
        device="cuda"
    ):
        super(Encoder, self).__init__()
        self.qkv_generator = PseudoGenerator(img_size=img_size, device=device)
        self.num_heads = num_heads
        self.dim = dim
        self.freeze_module(self.qkv_generator)

    def freeze_module(self, module):
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
        device="cuda",
        **kwargs,
    ):
        super().__init__(img_size=target_shp, device=device)

        self.num_heads = num_heads
        self.dim = dim
        self.target_shp = target_shp

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

        # Mask head: binary mask logits with 2 channels
        self.mask_head = nn.Conv2d(
            self.dim, 2, kernel_size=1
        )

        # Centroid head: heatmap
        self.center_head = nn.Conv2d(
            self.dim, 1, kernel_size=1
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
        feat_1_0, feat_2_0, _, _ = super().forward(img_1, img_2)
        
        # cross attention on (row, col)
        x_1_0, x_2_0 = self.apply_cross_attention(
            self, feat_1_0, feat_2_0, self.ca0s
        )
        x = torch.concatenate([x_1_0, x_2_0], dim=1)
        x = self.relu(self.conv1(x))
        
        cd_mask = self.relu(self.mask_head(x))  # (B, 2, H', W')
        center_preds = self.center_head(x)  # (B, 1, H', W')

        # Upsample to target size
        cd_mask_up = self.upsample(cd_mask)    # (B, 2, H, W)
        center_preds_up = self.upsample(center_preds)  # (B, 1, H, W)
        heatmap = torch.sigmoid(center_preds_up)  # (B,1,H,W)

        # ### reverse order ###
        feat_1_0, feat_2_0, _,_ = super().forward(img_2, img_1)

        # cross attention on (row, col)
        x_1_0, x_2_0 = self.apply_cross_attention(
            self, feat_1_0, feat_2_0, self.ca0s
        )
        x_reverse = torch.concatenate([x_1_0, x_2_0], dim=1)
        x_reverse = self.relu(self.conv1(x_reverse))
        cd_reverse = self.relu(self.mask_head(x_reverse))
        center_reverse = self.center_head(x_reverse)
        
        cd_reverse_mask_up = self.upsample(cd_reverse)
        center_reverse_up = self.upsample(center_reverse)
        heatpmap_reverse = torch.sigmoid(center_reverse_up)
        
        return {
            'mask': cd_mask_up,
            'mask_r': cd_reverse_mask_up,
            'heatmap': heatmap.detach(),
            'heatmap_r': heatpmap_reverse.detach()
        }