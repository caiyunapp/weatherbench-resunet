import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import sys

from leibniz.unet.base import UNet
from leibniz.unet.warp import WarpBottleneck
from leibniz.nn.activation import CappingRelu

sys.path.append('../')
from config import configs

def get_constants():
    cons_path = '/mnt/data02/mengqy/weather_bench/constants/constants_5.625deg.nc'
    cons = xr.open_dataset(cons_path).transpose('lat', 'lon')
    lsm = np.array(cons.lsm.data)
    lsm = (lsm- np.mean(lsm)) / np.std(lsm, ddof=1)
    orography = np.array(cons.orography.data)
    orography = (orography- np.mean(orography)) / np.std(orography, ddof=1)
    constants = np.stack((lsm, orography), axis=0)
    return torch.from_numpy(constants)

class WarpModel(nn.Module):
    def __init__(self):
        super().__init__()
        if configs.add_constants:
            self.constants = get_constants()
        self.unet = UNet(configs.input_dim,
                         configs.output_dim,
                         block=WarpBottleneck,
                         relu=CappingRelu(),
                         layers=4,
                         ratio=0,
                         vblks=[2, 2, 2, 2],
                         hblks=[0, 0, 0, 0],
                         scales=[-1, -1, -1, -1],
                         factors=[1, 1, 1, 1],
                         spatial=(32, 64),
                         normalizor='batch',
                         padding=None,
                         final_normalized=False)
    
    def add_constants(self, x):
        constants = self.constants.expand(x.shape[0], -1, -1, -1)
        x = torch.cat((x, constants.to(x.device)), dim=1)
        return x

    def forward(self, x):
        if configs.add_constants:
            x = self.add_constants(x)
        output = self.unet(x)
        return output

mdoel = WarpModel()