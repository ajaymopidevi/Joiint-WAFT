import os
import argparse
import torch
from algorithms.joint_waft import JointWAFT
from bridgedepth.config import get_cfg

def profile_joint_waft(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointWAFT(cfg).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    B, C, H, W = 1, 3, args.height, args.width
    sample = {
        "img1_t": torch.randn(B, C, H, W, device=device),
        "img2_t": torch.randn(B, C, H, W, device=device),
        "img1_t1": torch.randn(B, C, H, W, device=device),
    }

    # Profile FLOPs / MACs
    forward_flops = 0
    forward_macs = 0
    if device.type == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                output = model(sample)
        
        events = prof.events()
        forward_flops = sum([int(evt.flops) for evt in events])
        forward_macs = forward_flops / 2

        # Profile Peak VRAM
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(10): # Warmup & run
                output = model(sample)
        torch.cuda.synchronize()

        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        with torch.no_grad():
            output = model(sample)
        peak_allocated = 0
        peak_reserved = 0

    print("=================================================================")
    print("               Joint-WAFT Single Sample Profiling                ")
    print("=================================================================")
    print(f" Resolution:           {W}x{H}")
    print(f" Total Parameters:     {total_params / 1e6:.2f} M ({total_params:,} params)")
    print(f" Trainable Parameters: {trainable_params / 1e6:.2f} M ({trainable_params:,} params)")
    if device.type == "cuda":
        print(f" Forward FLOPs:        {forward_flops / 1e9:.2f} GFLOPs ({forward_flops / 1e12:.3f} TFLOPs)")
        print(f" Forward MACs:         {forward_macs / 1e9:.2f} GMACs")
        print(f" Peak Allocated VRAM:  {peak_allocated:.2f} MB ({peak_allocated / 1024:.2f} GB)")
        print(f" Peak Reserved VRAM:   {peak_reserved:.2f} MB ({peak_reserved / 1024:.2f} GB)")
    else:
        print(" CUDA not available; FLOPs/VRAM profiling requires GPU.")
    print("=================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/joint_spring.yaml", help="config file path")
    parser.add_argument("--height", type=int, default=540, help="image height (e.g. 540 for half Spring, 1080 for full)")
    parser.add_argument("--width", type=int, default=960, help="image width (e.g. 960 for half Spring, 1920 for full)")
    args = parser.parse_args()
    profile_joint_waft(args)
