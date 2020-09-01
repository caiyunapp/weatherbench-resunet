import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from config import configs
from costom import CubeSphereConv2D, CubeSpherePadding2D
from leibniz.unet.base import UNet
from leibniz.unet.senet import SELayer
from leibniz.unet.warp import BilinearWarpingScheme
from leibniz.nn.activation import CappingRelu

s=(0,-1,-1)

def get_constants():
    lsm_data = xr.load_dataset(f'/mnt/data02/mengqy/weather_bench/cubesphere/constants/lsm.nc')
    lsm = np.array(lsm_data.lsm)
    lsm = (lsm- np.mean(lsm)) / np.std(lsm, ddof=1)
    orography_data = xr.load_dataset(f'/mnt/data02/mengqy/weather_bench/cubesphere/constants/orography.nc')
    orography = np.array(orography_data.orography)
    orography = (orography- np.mean(orography)) / np.std(orography, ddof=1)
    constants = np.stack((lsm, orography), axis=0)
    return torch.from_numpy(constants)

class WarpLayer(nn.Module):
    def __init__(self, channel):
        super(WarpLayer, self).__init__()
        self.warp = BilinearWarpingScheme()
        self.se = SELayer(channel // 2)

    def forward(self, x):
        sz = x.size()
        u = x[:, 0::4]
        v = x[:, 1::4]
        y = x[:, 2::4]
        z = x[:, 3::4]
        ws = torch.cat((u, v), dim=1)
        ds = torch.cat((y, z), dim=1)
        if len(sz) == 3:
            raise Exception('Unimplemented')
        if len(sz) == 4:
            pst = self.warp(ds, ws)
            att = self.se(pst)
        if len(sz) == 5:
            pst = self.warp(ds[:,:,0], ws[:,:,0]).reshape(sz[0], -1, 1, sz[3], sz[4])
            for i in range(1,sz[2]):
                pst = torch.cat((pst, self.warp(ds[:,:,i], ws[:,:,i]).reshape(sz[0], -1, 1, sz[3], sz[4])), dim=2)
            att = self.se(pst)

        return torch.cat([ws, ds * att], dim=1)

class WarpBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(WarpBottleneck, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = conv(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(dim // 4, dim, kernel_size=1, bias=False)
        self.wp = WarpLayer(dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.wp(y)
        y = x + y

        return y

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

class WarpModel(nn.Module):
    def __init__(self):
        super().__init__()
        if configs.add_constants:
            self.constants = get_constants()
        self.unet = ReUNet(configs.input_dim, configs.output_dim, block=WarpBottleneck, relu=CappingRelu(), layers=4, ratio=0,
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

