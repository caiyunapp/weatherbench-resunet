import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from config import configs
from costom import CubeSphereConv2D, CubeSpherePadding2D
from leibniz.unet.base import UNet
from leibniz.unet.hyperbolic import HyperBottleneck
from leibniz.nn.activation import CappingRelu

s=(0,-1,-1)

def get_constants():
    lsm_data = xr.load_dataset(f'/data/weatherbench/CubeSphere/constants/lsm.nc')
    lsm = np.array(lsm_data.lsm)
    lsm = (lsm- np.mean(lsm)) / np.std(lsm, ddof=1)
    orography_data = xr.load_dataset(f'/data/weatherbench/CubeSphere/constants/orography.nc')
    orography = np.array(orography_data.orography)
    orography = (orography- np.mean(orography)) / np.std(orography, ddof=1)
    constants = np.stack((lsm, orography), axis=0)
    return torch.from_numpy(constants)

class ReUNet(UNet):
    def __init__(self, in_channels, out_channels, block=None, attn=None, relu=None, layers=4, ratio=2,
                 vblks=None, hblks=None, scales=None, factors=None, spatial=(256, 256),
                 normalizor='batch', padding=None, final_normalized=True):
        super().__init__(in_channels, out_channels, block=block, attn=None, relu=relu, layers=layers, ratio=ratio,
                 vblks=vblks, hblks=hblks, scales=scales, factors=factors, spatial=spatial,
                 normalizor=normalizor, padding=padding, final_normalized=final_normalized)

    def get_conv_for_prepare(self):
        if self.dim == 1:
            conv = nn.Conv1d
        elif self.dim == 2:
            conv = nn.Conv2d
        elif self.dim == 3:
            conv = CubeSphereConv2D
        else:
            raise ValueError('dim %d is not supported!' % self.dim)
        return conv

    def get_conv_for_transform(self):
        if self.dim == 1:
            conv = nn.Conv1d
        elif self.dim == 2:
            conv = nn.Conv2d
        elif self.dim == 3:
            conv = CubeSphereConv2D
        else:
            raise ValueError('dim %d is not supported!' % self.dim)
        return conv

class HyperModel(nn.Module):
    def __init__(self):
        super().__init__()
        if configs.add_constants:
            self.constants = get_constants()
        self.unet = ReUNet(configs.input_dim, configs.output_dim, block=HyperBottleneck, relu=CappingRelu(), layers=4, ratio=0,
                vblks=[2, 2, 2, 2], hblks=[0, 0, 0, 0], scales=[s, s, s, s], factors=[1, 1, 1, 1], spatial=(6, 48, 48),
                normalizor='batch', padding=CubeSpherePadding2D(1), final_normalized=False)
    
    def add_constants(self, x):
        constants = self.constants.expand(x.shape[0], -1, -1, -1, -1)
        x = torch.cat((x, constants.to(x.device)), dim=1)
        return x

    def forward(self, x):
        if configs.add_constants:
            x = self.add_constants(x)
        output = self.unet(x)
        return output

