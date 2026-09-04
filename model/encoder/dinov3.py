import torch
import torch.nn as nn
import timm
import numpy as np
import torchvision
import torch.nn.functional as F

import sys
import os
from timm.layers import Mlp
from peft import LoraConfig, get_peft_model
from einops import rearrange

from model.layers.dpt import UpsampleFeats, ProjFeats
from model.layers.block import resconv

class DINOv3Encoder(nn.Module):
    def __init__(self, model_name='vits', alpha=None, r=None):
        super().__init__()
        self.dpt_configs = {
            'vit7b': {'encoder': 'vit_7b_patch16_dinov3.lvd1689m', 'n_layers': 40, "dim": 4096},
            'vith+': {'encoder': 'vit_huge_plus_patch16_dinov3.lvd1689m','n_layers': 32, "dim": 1280},
            'vitl': {'encoder': 'vit_large_patch16_dinov3.lvd1689m', 'n_layers': 24, "dim": 1024},
            'vitb': {'encoder': 'vit_base_patch16_dinov3.lvd1689m', 'n_layers': 12, "dim": 768},
            'vits': {'encoder': 'vit_small_patch16_dinov3.lvd1689m', 'n_layers': 12, "dim": 384},
        }
        self.model_name = model_name
        self.dim = self.dpt_configs[model_name]['dim']
        self.idx = [(i + 1) * self.dpt_configs[model_name]['n_layers'] // 4 - 1 for i in range(4)]
        self.out_c = [(int)(self.dim * (2 ** (i - 3))) for i in range(4)]
        self.output_dim = self.out_c[0]
        vit = timm.create_model(
            self.dpt_configs[model_name]['encoder'],
            pretrained=True,
            features_only=True,
            out_indices=self.idx
        )
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["qkv", "proj"]
        )
        self.encoder = get_peft_model(vit, lora_config)
        self.fmap_proj = ProjFeats(self.dim, self.out_c, lvl=-3)
        self.fmap_upsample = UpsampleFeats(self.output_dim, self.out_c)
        self.hidden_proj = ProjFeats(self.dim*2, self.out_c, lvl=-3)
        self.hidden_upsample = UpsampleFeats(self.output_dim, self.out_c)

    def forward(self, imgs):
        """
        imgs: [B, N, C, H, W]
        Returns:
            fmaps: [B, N, C_out, H/2, W/2]
            hidden: [B, C_out, H/2, W/2]
        """
        B, N, _, H, W = imgs.shape
        flat_imgs = imgs.reshape(B*N, _, H, W)
        feats = self.encoder(flat_imgs)

        fmap_feats = self.fmap_proj(feats)
        fmap_feats = self.fmap_upsample(fmap_feats)
        fmaps = rearrange(fmap_feats[0], '(b n) c h w -> b n c h w', b=B, n=N)

        ref_feats = [rearrange(x, '(b n) c h w -> b n c h w', b=B, n=N)[:, :2] for x in feats]
        hidden_feats = [rearrange(x, 'b n c h w -> b (n c) h w') for x in ref_feats]
        hidden_feats = self.hidden_proj(hidden_feats)
        hidden_feats = self.hidden_upsample(hidden_feats)
        hidden = hidden_feats[0]

        return fmaps, hidden