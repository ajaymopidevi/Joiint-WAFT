import torch
import math
from torch import nn
from torch.nn import functional as F
from model.utils import disp_warp, flow_warp

def mixlap_loss(disp_preds, info_preds, target, loss_gamma=0.9, max_disp=192):
    disp_gt = target['disp']
    valid = target['valid_disp'] if 'valid_disp' in target else target['valid']
    valid = ((valid >= 0.5) & (disp_gt < max_disp))
    if valid.sum() == 0:
        return torch.sum(disp_preds[0]) * 0.0
    
    lap_loss = 0.0
    n_predictions = len(disp_preds)
    for i in range(n_predictions):
        i_weight = loss_gamma**(n_predictions - i - 1)
        disp_pred = disp_preds[i].squeeze(1)
        weights = info_preds[i][:, :2]
        # laplace likelihood loss
        log_b = torch.clamp(info_preds[i][:, 2], min=0, max=10)
        term0 = (disp_gt - disp_pred).abs() * torch.exp(-log_b) + log_b + math.log(2)
        term1 = (disp_gt - disp_pred).abs() + math.log(2)
        lap_term = torch.logsumexp(weights[:, :2], dim=1) - torch.logsumexp(weights[:, :2] - torch.stack([term0, term1], dim=1), dim=1)
        lap_loss += i_weight * lap_term[valid.bool() & ~torch.isnan(lap_term)].mean()

    return lap_loss

def disp_init_loss(prob_init, target, max_disp=192):
    disp_gt = target['disp']
    valid = target['valid_disp'] if 'valid_disp' in target else target['valid']
    n_bins = prob_init.shape[1]
    disp_gt = disp_gt.to(prob_init.device)
    valid = valid.to(prob_init.device)
    valid = ((valid >= 0.5) & (disp_gt < max_disp))
    if valid.sum() == 0:
        return torch.tensor(0.0, device=disp_gt.device)
    disp_gt = torch.clamp(disp_gt, min=0, max=max_disp-1)
    idx_bins = torch.linspace(0, max_disp, n_bins, device=disp_gt.device, dtype=disp_gt.dtype).view(1, n_bins, 1, 1)
    label = F.softmax(-(idx_bins - disp_gt.unsqueeze(1)).abs(), dim=1)
    prob = F.softmax(prob_init, dim=1)
    kl_loss = -(torch.log(torch.clamp(prob, min=1e-6)) * label).sum(dim=1)
    return kl_loss[valid.bool() & ~torch.isnan(kl_loss)].mean()

def sequence_flow_loss(flow_preds, target, loss_gamma=0.8, max_flow=400.0):
    flow_gt = target['flow']
    valid = target['valid_flow'] if 'valid_flow' in target else torch.ones_like(flow_gt[:, 0])
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    if valid.sum() == 0:
        return torch.sum(flow_preds[0]) * 0.0
    
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = loss_gamma ** (n_predictions - i - 1)
        diff = (flow_preds[i] - flow_gt).abs()
        loss_i = diff.sum(dim=1) # L1 flow error per pixel
        flow_loss += i_weight * loss_i[valid.bool() & ~torch.isnan(loss_i)].mean()
    return flow_loss

def cycle_consistency_loss(disp_pred, flow_pred, disp_next_pred=None, flow_right_pred=None):
    """
    Computes 4-view geometric loop consistency in 3D/pixel space if multiple views are predicted.
    """
    if disp_next_pred is None or flow_right_pred is None:
        return torch.tensor(0.0, device=disp_pred.device)
    
    # Warped next disparity onto time t:
    warped_disp_next = flow_warp(disp_next_pred.unsqueeze(1), flow_pred)
    # Warped right flow onto left camera:
    warped_flow_right = disp_warp(flow_right_pred, disp_pred.unsqueeze(1))
    
    # Cyclic loop residual
    cycle_residual = (flow_pred[:, 0:1] - (warped_flow_right[:, 0:1] + disp_pred.unsqueeze(1) - warped_disp_next)).abs()
    return cycle_residual.mean()

class JointWAFTCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.disp_weight = getattr(cfg.JOINT, 'DISP_WEIGHT', 1.0)
        self.flow_weight = getattr(cfg.JOINT, 'FLOW_WEIGHT', 1.0)
        self.cycle_weight = getattr(cfg.JOINT, 'CYCLE_WEIGHT', 0.1)

    def forward(self, outputs, targets, log=False):
        loss_dict = {}
        metrics = {}

        # 1. Disparity Metrics & Losses
        if 'disp' in targets and 'disp_pred' in outputs:
            disp_gt = targets['disp']
            valid_disp = (targets['valid_disp'] >= 0.5 if 'valid_disp' in targets else targets['valid'] >= 0.5).to(disp_gt.device)
            epe_disp = (outputs['disp_pred'] - disp_gt).abs()
            metrics['Disp_EPE'] = epe_disp[valid_disp].mean()
            metrics['Disp_D1'] = ((epe_disp[valid_disp] > 3) & (epe_disp[valid_disp] / disp_gt[valid_disp] > 0.05)).float().mean().item()

            l_disp_init = disp_init_loss(outputs['disp_init'], targets, max_disp=getattr(self.cfg.WAFT, 'MAX_DISP', 192))
            l_disp_seq = mixlap_loss(outputs['delta_disp_preds'], outputs['disp_info_preds'], targets, max_disp=getattr(self.cfg.WAFT, 'MAX_DISP', 192))
            
            loss_dict['loss_disp_init'] = l_disp_init
            loss_dict['loss_disp_seq'] = l_disp_seq
            loss_dict['loss_disp'] = self.disp_weight * (l_disp_init + l_disp_seq)

        # 2. Flow Metrics & Losses
        if 'flow' in targets and 'flow_pred' in outputs and outputs['flow_pred'] is not None:
            flow_gt = targets['flow']
            valid_flow = (targets['valid_flow'] >= 0.5 if 'valid_flow' in targets else torch.ones_like(flow_gt[:, 0]).bool()).to(flow_gt.device)
            epe_flow = torch.norm(outputs['flow_pred'] - flow_gt, p=2, dim=1)
            metrics['Flow_EPE'] = epe_flow[valid_flow].mean()
            metrics['Flow_1px'] = (epe_flow[valid_flow] < 1.0).float().mean().item()

            l_flow_seq = sequence_flow_loss(outputs['delta_flow_preds'], targets)
            loss_dict['loss_flow_seq'] = l_flow_seq
            loss_dict['loss_flow'] = self.flow_weight * l_flow_seq

        # 3. Geometric Cycle Regularization (Optional)
        if 'disp_next_pred' in outputs and 'flow_right_pred' in outputs:
            l_cycle = cycle_consistency_loss(outputs['disp_pred'], outputs['flow_pred'], outputs['disp_next_pred'], outputs['flow_right_pred'])
            loss_dict['loss_cycle'] = self.cycle_weight * l_cycle
            metrics['Loss_Cycle'] = l_cycle.item()

        total_loss = sum(v for k, v in loss_dict.items() if k in ['loss_disp', 'loss_flow', 'loss_cycle'])
        loss_dict['total_loss'] = total_loss

        return loss_dict, metrics

def build_criterion(cfg):
    if cfg.ALGORITHM in ["joint_waft", "joint"]:
        return JointWAFTCriterion(cfg)
    else:
        raise ValueError(f"Unknown algorithm: {cfg.ALGORITHM}")
