import torch
import torch.nn as nn
import timm
import numpy as np
import torchvision
import torch.nn.functional as F

import sys
import os
from peft import LoraConfig, get_peft_model
from einops import rearrange
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from model.layers.dpt import UpsampleFeats, ProjFeats

DEPTH_ANYTHING_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

class DAv2Encoder(nn.Module):
    def __init__(self, model_name='vits', alpha=None, r=None):
        super().__init__()
        self.model_name = model_name
        depth_anything = DepthAnythingV2(**DEPTH_ANYTHING_CONFIGS[model_name])
        if os.path.exists(f'depth-anything-ckpts/depth_anything_v2_{model_name}.pth'):
            depth_anything.load_state_dict(torch.load(f'depth-anything-ckpts/depth_anything_v2_{model_name}.pth', map_location='cpu'))
        else:
            print(f"Warning: checkpoint for depth anything v2 {model_name} not found, using random weights.")
        self.dpt_configs = {
            'vitl': {'n_layers': 24, "dim": 1024},
            'vitb': {'n_layers': 12, "dim": 768},
            'vits': {'n_layers': 12, "dim": 384},
        }
        self.dim = self.dpt_configs[model_name]['dim']
        self.idx = [(i + 1) * self.dpt_configs[model_name]['n_layers'] // 4 - 1 for i in range(4)]
        self.out_c = [(int)(self.dim * (2 ** (i - 3))) for i in range(4)]
        self.output_dim = self.out_c[0]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["qkv", "proj"]
        )
        self.encoder = get_peft_model(depth_anything.pretrained, lora_config)
        self.fmap_proj = ProjFeats(self.dim, self.out_c, lvl=-3)
        self.fmap_upsample = UpsampleFeats(self.output_dim, self.out_c)
        self.hidden_proj = ProjFeats(self.dim*2, self.out_c, lvl=-3)
        self.hidden_upsample = UpsampleFeats(self.output_dim, self.out_c) 

    def forward(self, imgs):
        """
        imgs: [B, N, C, H, W] where N can be 2, 3, or 4
        Returns:
            fmaps: [B, N, C_out, H/2, W/2]
            hidden: [B, C_out, H/2, W/2] (context from reference frame pair img1 and img2)
        """
        B, N, _, H, W = imgs.shape
        h, w = H // 16, W // 16
        flat_imgs = imgs.reshape(B*N, _, H, W)
        
        flat_imgs = F.interpolate(flat_imgs, (h*14, w*14), mode='bilinear', align_corners=True)
        feats = self.encoder.get_intermediate_layers(flat_imgs, self.idx, return_class_token=True)
        fmap_feats = [rearrange(x[0], 'bn (h w) c -> bn c h w', h=h, w=w) for x in feats]
        fmap_feats = self.fmap_proj(fmap_feats)
        fmap_feats = self.fmap_upsample(fmap_feats)
        fmaps = rearrange(fmap_feats[0], '(b n) c h w -> b n c h w', b=B, n=N)

        # Context features derived from the first two reference frames (e.g. left_t and right_t or left_t and left_t+1)
        ref_feats = [rearrange(x[0], '(b n) (h w) c -> b n c h w', b=B, n=N, h=h, w=w)[:, :2] for x in feats]
        hidden_feats = [rearrange(x, 'b n c h w -> b (n c) h w') for x in ref_feats]
        hidden_feats = self.hidden_proj(hidden_feats)
        hidden_feats = self.hidden_upsample(hidden_feats)
        hidden = hidden_feats[0]

        return fmaps, hidden