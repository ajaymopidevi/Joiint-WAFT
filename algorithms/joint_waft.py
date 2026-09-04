import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp
from einops import rearrange

from model.iterative import fetch_iterative_module
from model.encoder import fetch_feature_encoder
from model.utils import Padder, disp_warp, flow_warp, gaussian_weights

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
    for p in module.buffers():
        p.requires_grad = False

class JointWAFT(nn.Module):
    """
    Joint-WAFT: Shared Backbone Multi-Task Architecture for Joint Stereo Disparity and Optical Flow.
    Input sequence contains stereo video frames: Left_t, Right_t, Left_{t+1}, (optional Right_{t+1}).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.WAFT.ITERATIVE_MODULE.TASK if hasattr(cfg.WAFT, 'ITERATIVE_MODULE') else ['delta']*8
        self.iters = len(self.task)
        self.n_bins = int(cfg.WAFT.LOSS[0].split('_')[-1]) + 1 if hasattr(cfg.WAFT, 'LOSS') else 64
        self.grad_checkpointing = getattr(cfg.SOLVER, 'GRADIENT_CHECKPOINTING', False) if hasattr(cfg, 'SOLVER') else False
        
        # 1. Shared Feature Encoder
        self.encoder, self.enc_dim, self.factor = fetch_feature_encoder(cfg.WAFT.FEATURE_ENCODER)
        self.hidden_dim = self.enc_dim

        # 2. Task-specific feature adapters to prevent negative transfer
        self.disp_feat_adapter = nn.Sequential(
            nn.Conv2d(self.enc_dim, self.enc_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.enc_dim, self.enc_dim, kernel_size=1)
        )
        self.flow_feat_adapter = nn.Sequential(
            nn.Conv2d(self.enc_dim, self.enc_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.enc_dim, self.enc_dim, kernel_size=1)
        )

        # 3. Disparity Head Modules (Stereo)
        self.prop_decoder = fetch_iterative_module(cfg.WAFT.ITERATIVE_MODULE.PROP_ITER, input_dim=self.hidden_dim)
        self.prop_proj = Mlp(self.enc_dim * 2, self.hidden_dim, self.hidden_dim, use_conv=True)
        self.disp_decoder = fetch_iterative_module(cfg.WAFT.ITERATIVE_MODULE.DELTA_ITER, input_dim=self.hidden_dim)
        # Inputs: fmap1_disp (enc_dim) + warped_fmap2_disp (enc_dim) + net_disp (hidden_dim) + disp (1) + flow_guidance (2)
        self.disp_proj = Mlp(self.enc_dim * 2 + self.hidden_dim + 1 + 2, self.hidden_dim, self.hidden_dim, use_conv=True)
        
        self.max_disp = getattr(cfg.WAFT, 'MAX_DISP', 192)
        self.disp_mask_head = Mlp(self.hidden_dim, self.hidden_dim, 4 * 9, use_conv=True)
        self.disp_dist_head = Mlp(self.hidden_dim, self.hidden_dim, 4, use_conv=True)
        self.delta_disp_head = Mlp(self.hidden_dim, self.hidden_dim, 1, use_conv=True)
        self.prop_mask_head = Mlp(self.hidden_dim, self.hidden_dim, 4 * 9, use_conv=True)
        self.prop_bins_head = Mlp(self.hidden_dim, self.hidden_dim, self.n_bins, use_conv=True)

        # 4. Flow Head Modules (Temporal Optical Flow)
        self.flow_decoder = fetch_iterative_module(cfg.WAFT.ITERATIVE_MODULE.DELTA_ITER, input_dim=self.hidden_dim)
        # Inputs: fmap1_flow (enc_dim) + warped_fmap_next_flow (enc_dim) + net_flow (hidden_dim) + flow (2) + disp_guidance (1)
        self.flow_proj = Mlp(self.enc_dim * 2 + self.hidden_dim + 2 + 1, self.hidden_dim, self.hidden_dim, use_conv=True)
        self.flow_mask_head = Mlp(self.hidden_dim, self.hidden_dim, 4 * 9, use_conv=True)
        self.flow_dist_head = Mlp(self.hidden_dim, self.hidden_dim, 4, use_conv=True)
        self.delta_flow_head = Mlp(self.hidden_dim, self.hidden_dim, 2, use_conv=True)

        # Cross-Task Recurrent Hidden State Fusion
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, kernel_size=1),
            nn.GELU()
        )

    def normalize_image(self, img):
        """ Normalizes batch images (B, C, H, W) [0-255] """
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img / 255.0).contiguous()

    def convex_upsample(self, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        return up_info.reshape(N, C, 2 * H, 2 * W)

    def forward(self, sample, disp_init=None, flow_init=None):
        """
        sample dictionary expecting:
            - 'img1_t': Left frame at time t (Reference)
            - 'img2_t': Right frame at time t (Stereo target)
            - 'img1_t1': Left frame at time t+1 (Flow target)
            - (optional) 'img2_t1': Right frame at time t+1 (for 4-view cycle consistency)
        """
        output = {}
        img1_t = self.normalize_image(sample['img1_t'])
        img2_t = self.normalize_image(sample['img2_t'])
        img1_t1 = self.normalize_image(sample['img1_t1'])

        frames = [img1_t, img2_t, img1_t1]
        has_right_t1 = 'img2_t1' in sample and sample['img2_t1'] is not None
        if has_right_t1:
            img2_t1 = self.normalize_image(sample['img2_t1'])
            frames.append(img2_t1)

        padder = Padder(img1_t.shape, factor=self.factor)
        frames_padded = [padder.pad(f) for f in frames]
        stacked_imgs = torch.stack(frames_padded, dim=1) # [B, N, 3, H, W]

        # 1. Single forward pass through shared backbone encoder
        fmaps, net_init = self.encoder(stacked_imgs)
        B, N, _, H, W = fmaps.shape

        fmap_L_t = fmaps[:, 0]
        fmap_R_t = fmaps[:, 1]
        fmap_L_t1 = fmaps[:, 2]

        # 2. Task adapters
        fmap_L_t_disp = self.disp_feat_adapter(fmap_L_t)
        fmap_R_t_disp = self.disp_feat_adapter(fmap_R_t)
        fmap_L_t_flow = self.flow_feat_adapter(fmap_L_t)
        fmap_L_t1_flow = self.flow_feat_adapter(fmap_L_t1)

        # 3. Disparity Initialization (Stereo initial prior via histogram bins)
        idx_bins_2x = torch.linspace(0, self.max_disp / 2, self.n_bins, device=fmap_L_t.device, dtype=fmap_L_t.dtype).view(1, self.n_bins, 1, 1)
        idx_bins_1x = torch.linspace(0, self.max_disp / 1, self.n_bins, device=fmap_L_t.device, dtype=fmap_L_t.dtype).view(1, self.n_bins, 1, 1)

        prop_hidden = self.prop_proj(torch.cat([fmap_L_t_disp, fmap_R_t_disp], dim=1))
        if self.training and self.grad_checkpointing:
            prop_hidden = torch.utils.checkpoint.checkpoint(self.prop_decoder, prop_hidden, use_reentrant=False)
        else:
            prop_hidden = self.prop_decoder(prop_hidden)
        prob_mask = 0.25 * self.prop_mask_head(prop_hidden)
        prob_bins = self.prop_bins_head(prop_hidden)
        prob_up = self.convex_upsample(prob_bins, prob_mask)
        output['disp_init'] = padder.unpad(prob_up)
        prob_bins = F.softmax(prob_bins, dim=1)
        disp = torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)

        if disp_init is not None:
            disp = padder.pad(disp_init.unsqueeze(1))
            disp = F.interpolate(disp, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

        # 4. Flow Initialization (Zero or given init)
        if flow_init is None:
            flow = torch.zeros(B, 2, H, W, device=fmap_L_t.device, dtype=fmap_L_t.dtype)
        else:
            flow = padder.pad(flow_init)
            flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

        net_disp = net_init
        net_flow = net_init

        delta_disp_preds = []
        delta_flow_preds = []
        disp_info_preds = []
        flow_info_preds = []

        # 5. Joint Warped Attention Iterative Refinement
        for itr in range(self.iters):
            disp = disp.detach()
            flow = flow.detach()

            # Cross-Task hidden state exchange
            fused_net = self.cross_fusion(torch.cat([net_disp, net_flow], dim=1))
            c_disp, c_flow = torch.chunk(fused_net, 2, dim=1)
            net_disp = net_disp + c_disp
            net_flow = net_flow + c_flow

            # A. Stereo Branch: 1D Epipolar Warping & Refinement
            warped_fmap_R_disp = disp_warp(fmap_R_t_disp, disp, padding_mode='zeros')
            disp_inp = torch.cat([fmap_L_t_disp, warped_fmap_R_disp, net_disp, disp, flow], dim=1)
            net_disp = self.disp_proj(disp_inp)
            if self.training and self.grad_checkpointing:
                net_disp = torch.utils.checkpoint.checkpoint(self.disp_decoder, net_disp, use_reentrant=False)
            else:
                net_disp = self.disp_decoder(net_disp)
            
            delta_disp = self.delta_disp_head(net_disp)
            disp_info = self.disp_dist_head(net_disp)
            disp_mask = 0.25 * self.disp_mask_head(net_disp)
            disp = disp + delta_disp

            disp_up = self.convex_upsample(disp * 2, disp_mask)
            disp_info_up = self.convex_upsample(disp_info, disp_mask)
            delta_disp_preds.append(disp_up)
            disp_info_preds.append(disp_info_up)

            # B. Flow Branch: 2D Motion Warping & Refinement
            warped_fmap_L_flow = flow_warp(fmap_L_t1_flow, flow, padding_mode='zeros')
            flow_inp = torch.cat([fmap_L_t_flow, warped_fmap_L_flow, net_flow, flow, disp], dim=1)
            net_flow = self.flow_proj(flow_inp)
            if self.training and self.grad_checkpointing:
                net_flow = torch.utils.checkpoint.checkpoint(self.flow_decoder, net_flow, use_reentrant=False)
            else:
                net_flow = self.flow_decoder(net_flow)

            delta_flow = self.delta_flow_head(net_flow)
            flow_info = self.flow_dist_head(net_flow)
            flow_mask = 0.25 * self.flow_mask_head(net_flow)
            flow = flow + delta_flow

            flow_up = self.convex_upsample(flow * 2, flow_mask)
            flow_info_up = self.convex_upsample(flow_info, flow_mask)
            delta_flow_preds.append(flow_up)
            flow_info_preds.append(flow_info_up)

        output['delta_disp_preds'] = [padder.unpad(d) for d in delta_disp_preds]
        output['disp_info_preds'] = [padder.unpad(i) for i in disp_info_preds]
        output['delta_flow_preds'] = [padder.unpad(f) for f in delta_flow_preds]
        output['flow_info_preds'] = [padder.unpad(i) for i in flow_info_preds]

        output['disp_pred'] = output['delta_disp_preds'][-1].squeeze(1) if len(delta_disp_preds) > 0 else torch.sum(F.softmax(output['disp_init'], dim=1) * idx_bins_1x, dim=1)
        output['flow_pred'] = output['delta_flow_preds'][-1] if len(delta_flow_preds) > 0 else None

        return output
