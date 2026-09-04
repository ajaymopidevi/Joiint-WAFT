import random
import numpy as np
from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torchvision.transforms import ColorJitter, functional, Compose
from .transforms import AdjustGamma

class JointStereoFlowAugmentor:
    """
    Joint Augmentor for Stereo Video Quad tuples:
    (img1_t, img2_t, img1_t1, (img2_t1), disp_t, flow_t, valid_disp, valid_flow)
    """
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.4, do_flip=False, saturation_range=[0.6, 1.4], gamma=[1, 1, 1, 1]):
        crop_size[0] = crop_size[0] // 8 * 8
        crop_size[1] = crop_size[1] // 8 * 8
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.5
        self.max_stretch = 0.1

        self.do_flip = do_flip
        self.v_flip_prob = 0.1

        self.photo_aug = Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14),
            AdjustGamma(*gamma)
        ])
        self.asymmetric_color_aug_prob = 0.2

    def color_transform(self, imgs):
        """ Photometric augmentation applied symmetrically or with mild asymmetry """
        if np.random.rand() < self.asymmetric_color_aug_prob:
            return [np.array(self.photo_aug(Image.fromarray(im)), dtype=np.uint8) for im in imgs]
        else:
            stack = np.concatenate(imgs, axis=0)
            stack = np.array(self.photo_aug(Image.fromarray(stack)), dtype=np.uint8)
            return np.split(stack, len(imgs), axis=0)

    def spatial_transform(self, imgs, disp, flow, valid_disp, valid_flow):
        ht, wd = imgs[0].shape[:2]
        
        # Random scale
        min_scale = np.maximum((self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd))
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob or min_scale > 1.0:
            imgs = [cv2.resize(im, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) for im in imgs]
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) * scale_x
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) * [scale_x, scale_y]
            valid_disp = cv2.resize(valid_disp.astype(np.float32), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) > 0.5
            valid_flow = cv2.resize(valid_flow.astype(np.float32), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) > 0.5

        # Random Crop
        cur_ht, cur_wd = imgs[0].shape[:2]
        y0 = np.random.randint(0, max(1, cur_ht - self.crop_size[0] + 1))
        x0 = np.random.randint(0, max(1, cur_wd - self.crop_size[1] + 1))

        imgs = [im[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for im in imgs]
        disp = disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_disp = valid_disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_flow = valid_flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return imgs, disp, flow, valid_disp, valid_flow

    def __call__(self, imgs, disp, flow, valid_disp, valid_flow):
        imgs = self.color_transform(imgs)
        imgs, disp, flow, valid_disp, valid_flow = self.spatial_transform(imgs, disp, flow, valid_disp, valid_flow)
        
        imgs = [np.ascontiguousarray(im) for im in imgs]
        disp = np.ascontiguousarray(disp)
        flow = np.ascontiguousarray(flow)
        valid_disp = np.ascontiguousarray(valid_disp)
        valid_flow = np.ascontiguousarray(valid_flow)

        return imgs, disp, flow, valid_disp, valid_flow
