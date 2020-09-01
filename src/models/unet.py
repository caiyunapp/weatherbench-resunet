import numpy as np
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cubesphere import CubeSpherePadding2D, CubeSphereConv2D


def get_constants():
    lsm_data = xr.load_dataset(
        f'/data/weatherbench/CubeSphere/constants/lsm.nc')
    lsm = np.array(lsm_data.lsm)
    lsm = (lsm - np.mean(lsm)) / np.std(lsm, ddof=1)
    orography_data = xr.load_dataset(
        f'/data/weatherbench/CubeSphere/constants/orography.nc')
    orography = np.array(orography_data.orography)
    orography = (orography - np.mean(orography)) / np.std(orography, ddof=1)
    constants = np.stack((lsm, orography), axis=0)
    return torch.from_numpy(constants)


class CubeSphereUNet2D(nn.Module):
    def __init__(self, configs, padding):
        super(CubeSphereUNet2D, self).__init__()
        self.layers_per_block = configs.layers_per_block
        hidden_channels = configs.hidden_channels
        self.num_blocks = len(configs.layers_per_block)
        self.configs = configs
        # add_constants
        if configs.add_constants:
            self.constants = get_constants()
        # skip connection start at layers.b{skip_connection_start}l{1}
        assert (self.num_blocks %
                2) == 1  # self.num_blocks should be an odd number
        self.skip_connection_start = (self.num_blocks + 1) // 2

        # cube_padding layer
        self.cube_padding = CubeSpherePadding2D(padding=padding)

        Cell = lambda in_channels, out_channels: CubeSphereConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=configs.kernel_size,
            bias=configs.bias)

        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(self.layers_per_block[b]):

                # number of input channels to the current layer
                if l == 0 and b == 0:
                    channels = configs.input_dim
                elif l == 0 and b < self.skip_connection_start:  # skip-connection
                    assert hidden_channels[b] == (2 * hidden_channels[b - 1])
                    channels = hidden_channels[b - 1]
                elif l == 0 and b >= self.skip_connection_start:
                    assert hidden_channels[b - 1] == (2 * hidden_channels[b])
                    channels = hidden_channels[b - 1]
                else:
                    channels = hidden_channels[b]

                lid = "b{}l{}".format(b, l)  # layer ID
                if l == (self.layers_per_block[b] -
                         1) and b >= (self.skip_connection_start -
                                      1):  # the last layer of the last block
                    self.layers[lid] = Cell(channels, hidden_channels[b] // 2)
                    self.layers["normalizor_" + lid] = nn.BatchNorm3d(
                        hidden_channels[b] // 2, affine=True)
                else:
                    self.layers[lid] = Cell(channels, hidden_channels[b])
                    self.layers["normalizor_" + lid] = nn.BatchNorm3d(
                        hidden_channels[b], affine=True)
            # the last layer of each block
            if b == (self.num_blocks - 1):
                self.layers[lid] = Cell(hidden_channels[b], hidden_channels[b])
                self.layers["normalizor_" + lid] = nn.BatchNorm3d(
                    hidden_channels[b], affine=True)
            elif b < (self.skip_connection_start - 1):
                self.layers["AvgPool{}".format(b)] = nn.AvgPool3d(
                    kernel_size=(1, 2, 2))
                # self.layers["AvgPool{}".format(b)] = nn.Upsample(scale_factor = (1,0.5,0.5), mode='trilinear')
            elif b >= (
                    self.skip_connection_start - 1
            ):  # if b >= (self.skip_connection_start - 1) and b < (self.num_blocks - 1):
                self.layers["upsample{}".format(b)] = nn.Upsample(
                    scale_factor=(1, 2, 2), mode='trilinear')
        #  in_channels of output layer
        if self.layers_per_block[-1] > 1:
            channels = hidden_channels[-1]
        else:  # if self.layers_per_block[-1] == 1
            channels = hidden_channels[-1] * 2

        self.layers["output"] = CubeSphereConv2D(channels,
                                                 configs.output_dim,
                                                 kernel_size=1,
                                                 padding=0,
                                                 bias=True)

    def swish(self, x):
        return x * F.sigmoid(x)

    def capping_relu(self, x):
        x = F.leaky_relu(x, negative_slope=0.1)
        return x.clamp(max=10)

    def add_constants(self, x):
        constants = self.constants.expand(x.shape[0], -1, -1, -1, -1)
        x = torch.cat((x, constants.to(x.device)), dim=1)
        return x

    def forward(self, input_x):
        if self.configs.add_constants:
            input_x = self.add_constants(input_x)
        queue = []  # previous outputs for skip connection
        for b in range(self.num_blocks):
            for l in range(self.layers_per_block[b]):
                lid = "b{}l{}".format(b, l)  # layer ID
                input_x = self.cube_padding(input_x)
                input_x = self.layers[lid](input_x)

                # activation1: capping_relu
                input_x = self.capping_relu(input_x)
                input_x = self.layers["normalizor_" + lid](input_x)
                # activation2: swish
                #input_x = self.swish(input_x)

            # the last layer
            if b < (self.skip_connection_start - 1):  # if Downsample
                queue.append(input_x)
                input_x = self.layers["AvgPool{}".format(b)](input_x)
            elif b == (self.num_blocks - 1):  # if output
                output = self.layers["output"](input_x)
            elif b >= (self.skip_connection_start - 1):  # if Upsample
                input_x = self.layers["upsample{}".format(b)](input_x)
                input_x = torch.cat([input_x, queue.pop()],
                                    dim=1)  # concat over the channels

        return output
