import torch
from algorithms.joint_waft import JointWAFT
from bridgedepth.config import get_cfg

def test_joint_waft():
    print("Testing Joint-WAFT forward pass...")
    cfg = get_cfg()
    cfg.merge_from_file("configs/joint_waft.yaml")
    cfg.WAFT.FEATURE_ENCODER.TYPE = "dav2"
    cfg.WAFT.FEATURE_ENCODER.ARCH = "vits"
    
    # Instantiate Joint-WAFT
    model = JointWAFT(cfg)
    model.eval()

    B, C, H, W = 1, 3, 256, 384
    # Dummy stereo video inputs (Left_t, Right_t, Left_t+1)
    sample = {
        'img1_t': torch.randint(0, 256, (B, C, H, W), dtype=torch.float32),
        'img2_t': torch.randint(0, 256, (B, C, H, W), dtype=torch.float32),
        'img1_t1': torch.randint(0, 256, (B, C, H, W), dtype=torch.float32),
    }

    with torch.no_grad():
        output = model(sample)

    print("Success! Forward output shapes:")
    print(" - Disparity Pred shape:", output['disp_pred'].shape)
    print(" - Flow Pred shape:     ", output['flow_pred'].shape)
    print(" - Number of disp iterations:", len(output['delta_disp_preds']))
    print(" - Number of flow iterations:", len(output['delta_flow_preds']))

if __name__ == '__main__':
    test_joint_waft()
