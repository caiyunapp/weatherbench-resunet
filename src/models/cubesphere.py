import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = None
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction + 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction + 1, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        sz = x.size()

        if len(sz) == 3:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1)
        if len(sz) == 4:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1, 1)
        if len(sz) == 5:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1, 1, 1)
        return x * y.expand_as(x)


class CubeSphereConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 flip_north_pole=True,
                 independent_north_pole=False):
        super(CubeSphereConv2D, self).__init__()

        self.flip_north_pole = flip_north_pole
        self.independent_north_pole = independent_north_pole

        self.conv_equator = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias)
        nn.init.xavier_uniform_(self.conv_equator.weight)
        nn.init.constant_(self.conv_equator.bias, 0)

        self.conv_pole = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=bias)
        nn.init.xavier_uniform_(self.conv_pole.weight)
        nn.init.constant_(self.conv_pole.bias, 0)

        if self.independent_north_pole:
            self.conv_north_pole = nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             dilation=dilation,
                                             bias=bias)
            nn.init.xavier_uniform_(self.conv_north_pole.weight)
            nn.init.constant_(self.conv_north_pole.bias, 0)

        self.se = SELayer(channel=out_channels)

    def forward(self, input):
        assert len(input.shape) == 5
        # Equator
        outputs_e = [self.conv_equator(input[:, :, f, :, :]) for f in range(4)]
        # South Pole
        outputs_s = self.conv_pole(input[:, :, 4, :, :])
        # North Pole
        if self.flip_north_pole:
            input[:, :, 5, :, :] = torch.flip(input[:, :, 5, :, :],
                                              dims=(-2, ))
        if self.independent_north_pole:
            outputs_n = self.conv_north_pole(input[:, :, 5, :, :])
        else:
            outputs_n = self.conv_pole(input[:, :, 5, :, :])
        if self.flip_north_pole:
            outputs_n = torch.flip(outputs_n, dims=(-2, ))
        outputs = torch.stack((outputs_e[0], outputs_e[1], outputs_e[2],
                               outputs_e[3], outputs_s, outputs_n),
                              dim=2)
        outputs = self.se(outputs)
        return outputs


class CubeSpherePadding2D(nn.Module):
    def __init__(self, padding):
        super(CubeSpherePadding2D, self).__init__()
        self.pad = padding

    def forward(self, t):
        pad = self.pad
        assert len(t.shape) == 5
        t = F.pad(t, (pad, pad, pad, pad, 0, 0), mode='constant', value=0)
        t[:, :, 0, :, -pad:] = t[:, :, 1, :, pad:2 * pad]
        t[:, :, 0, :, :pad] = t[:, :, 3, :, -2 * pad:-pad]
        t[:, :, 0, :pad, :] = t[:, :, 4, -2 * pad:-pad, :]
        t[:, :, 0, -pad:, :] = t[:, :, 5, pad:2 * pad, :]

        t[:, :, 1, :, -pad:] = t[:, :, 2, :, pad:2 * pad]
        t[:, :, 1, :, :pad] = t[:, :, 0, :, -2 * pad:-pad]
        t[:, :, 1, :pad, :] = torch.flip(t[:, :, 4, :,
                                           -2 * pad:-pad].transpose(-1, -2),
                                         dims=(-1, ))
        t[:, :, 1, -pad:, :] = torch.flip(t[:, :, 5, :,
                                            -2 * pad:-pad].transpose(-1, -2),
                                          dims=(-2, ))

        t[:, :, 2, :, -pad:] = t[:, :, 3, :, pad:2 * pad]
        t[:, :, 2, :, :pad] = t[:, :, 1, :, -2 * pad:-pad]
        t[:, :, 2, :pad, :] = torch.flip(t[:, :, 4, pad:2 * pad, :],
                                         dims=(-1, -2))
        t[:, :, 2, -pad:, :] = torch.flip(t[:, :, 5, -2 * pad:-pad, :],
                                          dims=(-1, -2))

        t[:, :, 3, :, -pad:] = t[:, :, 0, :, pad:2 * pad]
        t[:, :, 3, :, :pad] = t[:, :, 2, :, -2 * pad:-pad]
        t[:, :, 3, :pad, :] = torch.flip(t[:, :, 4, :,
                                           pad:2 * pad].transpose(-1, -2),
                                         dims=(-2, ))
        t[:, :, 3, -pad:, :] = torch.flip(t[:, :, 5, :,
                                            pad:2 * pad].transpose(-1, -2),
                                          dims=(-1, ))

        t[:, :, 4, :, -pad:] = torch.flip(t[:, :, 1,
                                            pad:2 * pad, :].transpose(-1, -2),
                                          dims=(-2, ))
        t[:, :, 4, :, :pad] = torch.flip(t[:, :, 3,
                                           pad:2 * pad, :].transpose(-1, -2),
                                         dims=(-1, ))
        t[:, :, 4, :pad, :] = torch.flip(t[:, :, 2, pad:2 * pad, :],
                                         dims=(-1, -2))
        t[:, :, 4, -pad:, :] = t[:, :, 0, pad:2 * pad, :]

        t[:, :, 5, :,
          -pad:] = torch.flip(t[:, :, 1, -2 * pad:-pad, :].transpose(-1, -2),
                              dims=(-1, ))
        t[:, :, 5, :, :pad] = torch.flip(t[:, :, 3,
                                           -2 * pad:-pad, :].transpose(-1, -2),
                                         dims=(-2, ))
        t[:, :, 5, :pad, :] = t[:, :, 0, -2 * pad:-pad, :]
        t[:, :, 5, -pad:, :] = torch.flip(t[:, :, 2, -2 * pad:-pad, :],
                                          dims=(-1, -2))
        return t
