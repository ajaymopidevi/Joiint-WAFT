import h5py
from os.path import splitext
import re
import numpy as np
from PIL import Image
from os.path import *
import torch.nn.functional as F
import json
import imageio
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
        os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = (disp > 0.0) & (disp < 1e3)
    return disp, valid

def readDispETH3D(file_name, nonocc=False):
    disp = readPFM(file_name).astype(np.float32)
    assert len(disp.shape) == 2
    if nonocc:
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        valid = nocc_pix & (disp > 0.0) & (disp < 1e3)
    else:
        valid = (disp > 0.0) & (disp < 1e3)
    return disp, valid

def readDispMiddlebury(file_name, nonocc=False):
    disp = readPFM(file_name).astype(np.float32)
    assert len(disp.shape) == 2
    if nonocc:
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        valid = nocc_pix & (disp > 0.0) & (disp < 1e3)
    else:
        valid = (disp > 0.0) & (disp < 1e3)
    return disp, valid

def readDispBooster(file_name, nonocc=False):
    disp = readPFM(file_name).astype(np.float32)
    assert len(disp.shape) == 2
    valid = (disp > 0.0) & (disp < 1e3)
    return disp, valid

def readDispInStereo2K(file_name):
    disp = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH) / 100.0
    valid = disp > 0
    return disp, valid

def readDispCREStereo(file_name):
    disp = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH) / 32.0
    valid = disp > 0
    return disp, valid

def readDispFallingThings(file_name):
    disp = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0
    return disp, valid

def readDispFSD(file_name, scale=1000):
    depth_uint8 = np.asarray(Image.open(file_name))
    depth_uint8 = depth_uint8.astype(float)
    disp = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
    valid = disp > 0
    return disp/float(scale), valid

def writeDispSpring(filename, disp):
    with h5py.File(filename, "w") as f:
        f.create_dataset("disparity", data=disp, compression="gzip", compression_opts=5)

def readDispSpring(file_name, half_res=True):
    with h5py.File(file_name, "r") as f:
        disp = f["disparity"][()]
    if half_res:
        disp = disp[::2, ::2]
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = (disp > 0.0) & (disp < 1e4)
    return disp, valid

def readFlowSpring(file_name, half_res=True):
    with h5py.File(file_name, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {file_name} does not have a 'flow' key.")
        flow = f["flow"][()]
    if half_res:
        flow = flow[::2, ::2]
    flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
    return flow, valid

def writeFlowSpring(filename, flow):
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)

def readDispTartanGround(file_name):
    depth_rgba = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4")
    depth = np.squeeze(depth, axis=-1)
    disp = 80.0 / (depth + 1e-6)
    valid = disp > 0
    return disp, valid

def readDispUnrealStereo4K(file_name):
    disp = np.load(file_name, mmap_mode='c')
    valid = disp > 0
    return disp, valid

def readDispWMGStereo(file_name):
    disp = np.load(file_name)
    valid = disp > 0
    return disp, valid

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext in ['.png', '.jpeg', '.ppm', '.jpg']:
        return Image.open(file_name)
    elif ext in ['.bin', '.raw']:
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.flo5':
        flow, _ = readFlowSpring(file_name, half_res=True)
        return flow
    elif ext == '.dsp5':
        disp, _ = readDispSpring(file_name, half_res=True)
        return disp
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'nmrf':
            self._pad = [0, pad_wd, pad_ht, 0]
        else:
            raise ValueError(f"Non recognized mode '{mode}'")

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
