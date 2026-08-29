"""
Initial pseudo-mask generation
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from einops import rearrange

from project_config import CHECKPOINT_DIR, require_file, resolve_device
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class PseudoGenerator(nn.Module):
    ### model & feature selection ###
    # Large:     48, feature layer: 24
    # Base_plus: 24, feature layer: 13, dim=448
    # Small:     16, feature layer: 8
    ##################################
    
    def __init__(self, feature_layer=24, embedding_layer=4, img_size=(512,512), backbone=None, device="auto", sam2_checkpoint=None):
        super(PseudoGenerator, self).__init__()
        logging.info('build initial pseudo-mask generator')
        device = resolve_device(device)
        self.feature_layer = feature_layer
        self.embedding_layer = embedding_layer
        
        sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = sam2_checkpoint or CHECKPOINT_DIR / "sam2.1_hiera_large.pt"
        sam2_checkpoint = require_file(sam2_checkpoint, "SAM2 checkpoint")
        sam_backbone = backbone or build_sam2(
            sam2_model_cfg,
            str(sam2_checkpoint),
            device=device,
            apply_postprocessing=False,
        )
        
        self.backbone = sam_backbone.image_encoder.trunk
        self.backbone.eval()
        
        self.img_size = img_size
        self.patch_size = 16
        
        
    def __forward(self, img, return_qkv=False):
        ## intercept feature facet
        qkv = self.backbone.get_intermediate_layers(
            img, n=32, return_qkv=return_qkv, lyr=self.feature_layer
        )
        return qkv

    def forward(self, input_t0, input_t1):
        with torch.no_grad():
            input_t0_qkv = self.__forward(input_t0, return_qkv=True)
            input_t1_qkv = self.__forward(input_t1, return_qkv=True)

            t0_key, t1_key = self._generate(input_t0_qkv, input_t1_qkv)
        
            # embeds_t0 = self.backbone(input_t0)
            # embeds_t1 = self.backbone(input_t1)
            
            # embed_t0 = embeds_t0[self.embedding_layer-1].permute(0,3,1,2)
            # embed_t1 = embeds_t1[self.embedding_layer-1].permute(0,3,1,2)
            
            # embed_t0 = F.interpolate(embed_t0, self.img_size, mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)
            # embed_t1 = F.interpolate(embed_t1, self.img_size, mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)
            
        return t0_key, t1_key, 0, 0
    
    def _generate(self, input_t0_qkv, input_t1_qkv):
        ## multi-head feature correlation
        B, L, _, N, C = input_t0_qkv.shape
        ### small ###
        # ([4, 1024, 3, 4, 96]) 
        # seperate qkv
        input_t0_key = input_t0_qkv[:, :, 1, :, :]
        input_t1_key = input_t1_qkv[:, :, 1, :, :]
        input_t0_qry = input_t0_qkv[:, :, 0, :, :]
        input_t1_qry = input_t1_qkv[:, :, 0, :, :]
        input_t0_val = input_t0_qkv[:, :, 2, :, :]
        input_t1_val = input_t1_qkv[:, :, 2, :, :]
        
        h = int(self.img_size[0] // self.patch_size)
        w = int(self.img_size[1] // self.patch_size)
        
        # reshape to 2D image space
        input_t0_key = rearrange(input_t0_key, 'b (h w) n c -> b h w (n c)', h=h, w=w)
        input_t1_key = rearrange(input_t1_key, 'b (h w) n c -> b h w (n c)', h=h, w=w)
        input_t0_qry = rearrange(input_t0_qry, 'b (h w) n c -> b h w (n c)', h=h, w=w)
        input_t1_qry = rearrange(input_t1_qry, 'b (h w) n c -> b h w (n c)', h=h, w=w)
        input_t0_val = rearrange(input_t0_val, 'b (h w) n c -> b h w (n c)', h=h, w=w)
        input_t1_val = rearrange(input_t1_val, 'b (h w) n c -> b h w (n c)', h=h, w=w)
        
        # l2 normalization
        input_t0_key = F.normalize(input_t0_key, p=2, dim=-1)
        input_t1_key = F.normalize(input_t1_key, p=2, dim=-1)
        input_t0_qry = F.normalize(input_t0_qry, p=2, dim=-1)
        input_t1_qry = F.normalize(input_t1_qry, p=2, dim=-1)
        input_t0_val = F.normalize(input_t0_val, p=2, dim=-1)
        input_t1_val = F.normalize(input_t1_val, p=2, dim=-1)
        
        return input_t0_key, input_t1_key
    
