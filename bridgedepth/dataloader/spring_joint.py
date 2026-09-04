import os
import os.path as osp
import glob
from glob import glob
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from ..utils import frame_utils, misc
from .base.easy_dataset import EasyDataset

class SpringJointDataset(EasyDataset, data.Dataset):
    """
    Spring Dataset for Joint Stereo Disparity and Optical Flow Training.
    Returns:
        img1_t: Left camera frame at time t
        img2_t: Right camera frame at time t
        img1_t1: Left camera frame at time t+1
        img2_t1: Right camera frame at time t+1 (optional for 4-view cycle consistency)
        disp: Disparity ground truth at time t (left camera)
        flow: Forward Optical Flow ground truth from t -> t+1 (left camera)
        valid_disp: Valid mask for disparity
        valid_flow: Valid mask for optical flow
    """
    def __init__(self, aug_params=None, root='datasets/spring', split='train', subsample_half=True, return_right_t1=True):
        super().__init__()
        self.aug_params = aug_params
        self.subsample_half = subsample_half
        self.return_right_t1 = return_right_t1
        self.split = split
        self.init_seed = False

        seq_root = os.path.join(root, split)
        if not os.path.exists(seq_root):
            # Try fallback to standard directory name if train_val / train
            fallback_root = os.path.join(root, 'train_val') if split == 'train' else os.path.join(root, 'train')
            if os.path.exists(fallback_root):
                seq_root = fallback_root
            else:
                raise ValueError(f"Spring directory does not exist: {seq_root}")

        self.seq_root = seq_root
        self.samples = []

        scenes = sorted(os.listdir(seq_root))
        for scene in scenes:
            scene_dir = os.path.join(seq_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            
            left_imgs = sorted(glob(os.path.join(scene_dir, "frame_left", "*.png")))
            num_frames = len(left_imgs)
            if num_frames < 2:
                continue

            # Index contiguous frame pairs (frame t -> frame t+1)
            for frame_idx in range(1, num_frames):
                img_L_t = os.path.join(scene_dir, "frame_left", f"frame_left_{frame_idx:04d}.png")
                img_R_t = os.path.join(scene_dir, "frame_right", f"frame_right_{frame_idx:04d}.png")
                img_L_t1 = os.path.join(scene_dir, "frame_left", f"frame_left_{frame_idx+1:04d}.png")
                img_R_t1 = os.path.join(scene_dir, "frame_right", f"frame_right_{frame_idx+1:04d}.png")

                disp_path = os.path.join(scene_dir, "disp1_left", f"disp1_left_{frame_idx:04d}.dsp5")
                flow_path = os.path.join(scene_dir, "flow_FW_left", f"flow_FW_left_{frame_idx:04d}.flo5")

                if os.path.exists(img_L_t) and os.path.exists(img_R_t) and os.path.exists(img_L_t1):
                    self.samples.append({
                        'img_L_t': img_L_t,
                        'img_R_t': img_R_t,
                        'img_L_t1': img_L_t1,
                        'img_R_t1': img_R_t1,
                        'disp_path': disp_path if os.path.exists(disp_path) else None,
                        'flow_path': flow_path if os.path.exists(flow_path) else None,
                        'scene': scene,
                        'frame_idx': frame_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            initial_seed = torch.initial_seed() % (2**31)
            if worker_info is not None:
                misc.seed_all_rng(initial_seed + worker_info.id)
                self.init_seed = True

        item = self.samples[index]

        img1_t = np.array(Image.open(item['img_L_t'])).astype(np.uint8)[..., :3]
        img2_t = np.array(Image.open(item['img_R_t'])).astype(np.uint8)[..., :3]
        img1_t1 = np.array(Image.open(item['img_L_t1'])).astype(np.uint8)[..., :3]

        if self.return_right_t1 and os.path.exists(item['img_R_t1']):
            img2_t1 = np.array(Image.open(item['img_R_t1'])).astype(np.uint8)[..., :3]
            imgs = [img1_t, img2_t, img1_t1, img2_t1]
        else:
            img2_t1 = None
            imgs = [img1_t, img2_t, img1_t1]

        # Read disparity and optical flow
        if item['disp_path'] is not None:
            disp, valid_disp = frame_utils.readDispSpring(item['disp_path'], half_res=self.subsample_half)
        else:
            H, W = (img1_t.shape[0]//2, img1_t.shape[1]//2) if self.subsample_half else img1_t.shape[:2]
            disp = np.zeros((H, W), dtype=np.float32)
            valid_disp = np.zeros((H, W), dtype=bool)

        if item['flow_path'] is not None:
            flow, valid_flow = frame_utils.readFlowSpring(item['flow_path'], half_res=self.subsample_half)
        else:
            H, W = (img1_t.shape[0]//2, img1_t.shape[1]//2) if self.subsample_half else img1_t.shape[:2]
            flow = np.zeros((H, W, 2), dtype=np.float32)
            valid_flow = np.zeros((H, W), dtype=bool)

        # Scale down RGB if ground truth is half resolution
        if self.subsample_half:
            imgs = [im[::2, ::2] for im in imgs]

        # Augmentations
        if self.aug_params is not None:
            imgs, disp, flow, valid_disp, valid_flow = self.aug_params(imgs, disp, flow, valid_disp, valid_flow)

        sample = {
            'img1_t': torch.from_numpy(imgs[0]).permute(2, 0, 1).float(),
            'img2_t': torch.from_numpy(imgs[1]).permute(2, 0, 1).float(),
            'img1_t1': torch.from_numpy(imgs[2]).permute(2, 0, 1).float(),
            'disp': torch.from_numpy(disp).float(),
            'flow': torch.from_numpy(flow).permute(2, 0, 1).float(),
            'valid_disp': torch.from_numpy(valid_disp).bool(),
            'valid_flow': torch.from_numpy(valid_flow).bool(),
            'valid': torch.from_numpy(valid_disp).bool()
        }

        if img2_t1 is not None:
            sample['img2_t1'] = torch.from_numpy(imgs[3]).permute(2, 0, 1).float()

        return sample
