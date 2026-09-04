import logging
import os
import argparse
import sys
import json
import wandb

import torch
import copy
from torch.utils.tensorboard import SummaryWriter

from algorithms.joint_waft import JointWAFT
from bridgedepth.config import export_model_config
from bridgedepth.dataloader import build_train_loader
from bridgedepth.loss import build_criterion
from bridgedepth.utils import misc
import bridgedepth.utils.dist_utils as comm
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils.launch import launch
from bridgedepth.utils.eval_disp import eval_disp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_args_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="configs/joint_waft.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action='store_true')
    parser.add_argument("--ckpt", default=None, help='path to the checkpoint file')
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine")
    parser.add_argument("--dist-url", default="auto", help="dist url for init process")
    parser.add_argument("--use-wandb", action='store_true', help="enable Weights & Biases logging")
    parser.add_argument("--tb-dir", default=None, help="directory for tensorboard event logs (defaults to checkpoint_dir/tb)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    return parser

def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=1e-4, eps=1e-8)
    return optimizer

def _setup(cfg, args):
    alg_name = args.config_file.split('/')[-1].split('.')[0]
    args.checkpoint_dir = f"ckpts/joint_waft/{alg_name}/{args.seed}"
    checkpoint_dir = args.checkpoint_dir
    if comm.is_main_process() and checkpoint_dir:
        misc.check_path(checkpoint_dir)

    rank = comm.get_rank()
    logger = setup_logger(checkpoint_dir, distributed_rank=rank, name='joint_waft')
    logger.info("Environment info:\n" + misc.collect_env_info())
    logger.info("Command line arguments: " + str(args))

    if comm.is_main_process() and checkpoint_dir:
        path = os.path.join(checkpoint_dir, "config.yaml")
        with open(path, 'w') as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    misc.seed_all_rng(None if args.seed < 0 else args.seed + rank)
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

def setup(args):
    from bridgedepth.config import get_cfg
    cfg = get_cfg()
    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    _setup(cfg, args)
    comm.setup_for_distributed(comm.is_main_process())
    return cfg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    cfg = setup(args)
    model = JointWAFT(cfg)
    model = model.to(torch.device("cuda"))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if comm.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    logger = logging.getLogger("joint_waft")
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")

    optimizer = build_optimizer(model_without_ddp, cfg)
    criterion = build_criterion(cfg)

    # TensorBoard & WandB logger initialization
    tb_writer = None
    if comm.is_main_process():
        tb_log_dir = args.tb_dir if args.tb_dir is not None else os.path.join(args.checkpoint_dir, "tb_logs")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")
        
        if args.use_wandb:
            data_name = "joint_waft"
            exp_name = args.config_file.split('/')[-1].split('.')[0] + f"-{args.seed}"
            wandb.init(project=data_name, name=exp_name)

    # Dataloaders
    train_loader, train_sampler = build_train_loader(cfg)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.SOLVER.BASE_LR,
        cfg.SOLVER.MAX_ITER + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear'
    )

    total_steps = 0
    epoch = 0
    logger.info('Start training Joint-WAFT')
    avg_dict = {}

    while total_steps < cfg.SOLVER.MAX_ITER:
        model.train()
        if comm.get_world_size() > 1:
            train_sampler.set_epoch(epoch)

        for i_batch, sample in enumerate(train_loader):
            sample = {k: v.to(torch.device("cuda")) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.SOLVER.MIX_PRECISION):
                result_dict = model(sample)
                loss_dict, metrics = criterion(result_dict, sample, log=True)
                losses = loss_dict['total_loss']

            for param in model_without_ddp.parameters():
                param.grad = None

            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()

            # Record metrics
            if comm.is_main_process():
                curr_lr = optimizer.param_groups[0]['lr']
                for k, v in loss_dict.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    meter = avg_dict.get(k, AverageMeter())
                    meter.update(val)
                    avg_dict[k] = meter

                for k, v in metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    meter = avg_dict.get(k, AverageMeter())
                    meter.update(val)
                    avg_dict[k] = meter

            total_steps += 1

            # Log to TensorBoard and Console periodically
            if total_steps % 20 == 0 and comm.is_main_process():
                logger.info(f"Step [{total_steps}/{cfg.SOLVER.MAX_ITER}] Loss: {losses.item():.4f} LR: {curr_lr:.6f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("train/lr", curr_lr, total_steps)
                    for k, meter in avg_dict.items():
                        tb_writer.add_scalar(f"train/{k}", meter.avg, total_steps)
                    
                    if args.use_wandb:
                        wandb_dict = {f"train/{k}": meter.avg for k, meter in avg_dict.items()}
                        wandb_dict["train/lr"] = curr_lr
                        wandb.log(wandb_dict, step=total_steps)

                    avg_dict = {}

            # Save Checkpoint
            if (total_steps % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or total_steps == cfg.SOLVER.MAX_ITER) and comm.is_main_process():
                ckpt_path = os.path.join(args.checkpoint_dir, f"step_{total_steps:06d}.pth")
                torch.save({
                    'step': total_steps,
                    'epoch': epoch,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'cfg': cfg
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            if total_steps >= cfg.SOLVER.MAX_ITER:
                break
        epoch += 1

    if tb_writer is not None:
        tb_writer.close()

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )