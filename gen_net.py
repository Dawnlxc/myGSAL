import torch
from torch import nn
import torch.nn.functional as F
import copy

import warnings
warnings.filterwarnings(action='ignore')

from torchvision import models


class DoubleConvBlock(nn.Module):
    '''
      (Conv-BN-Relu) *2
    '''

    def __init__(self,
                 in_channels,
                 out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    '''
      Conv-BN-Relu
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 dilation=1,
                 bias=False,
                 bn=True,
                 relu=True):
        super(ConvBlock, self).__init__()
        conv = [nn.Conv2d(in_channels, out_channels,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=padding,
                          groups=groups,
                          dilation=(dilation, dilation),
                          bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channels))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=dilation,
                      dilation=dilation,
                      groups=in_channels),
            ConvBlock(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class DSConv5x5(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels,
                      in_channels,
                      kernel_size=5,
                      stride=stride,
                      padding=2 * dilation,
                      dilation=dilation,
                      groups=in_channels),
            ConvBlock(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels // 2,
                                         out_channels=in_channels // 2,
                                         kernel_size=(2, 2),
                                         stride=(2, 2))
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_x = skip.size()[2] - x.size()[2]
        diff_y = skip.size()[3] - x.size()[3]

        x = F.pad(x, (diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2))
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 is_deepsup=False):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_deepsup = is_deepsup

        feats = list(models.vgg16_bn(pretrained=True).features.children())
        feats[0] = nn.Conv2d(self.in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Sequential(*feats[:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])

        self.skeleton = nn.ModuleList()
        for i in range(3):
            self.skeleton.append(UpsampleBlock(1024 // 2 ** i, 256 // 2 ** i))
        self.skeleton.append(UpsampleBlock(128, 64))

        # self.boundary = copy.deepcopy(self.skeleton)
        self.boundary = nn.ModuleList()
        for i in range(3):
            self.boundary.append(UpsampleBlock(1024 // 2 ** i, 256 // 2 ** i))
        self.boundary.append(UpsampleBlock(128, 64))

        # self.outc = nn.Conv2d(in_channels=64 * 3,
        #                       out_channels=out_channels,
        #                       kernel_size=(1, 1))
        self.ske_mask_conv = nn.Conv2d(in_channels=64,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1))

        self.bound_mask_conv = nn.Conv2d(in_channels=64,
                                         out_channels=out_channels,
                                         kernel_size=(1, 1))

        self.seg_mask_conv = nn.Conv2d(in_channels=64,
                                       out_channels=out_channels,
                                       kernel_size=(1, 1))

        self.fusion_conv = DSConv3x3(in_channels=64 * 2,
                                     out_channels=64)

        self.act = nn.Sigmoid()

        # self.seg_heads = nn.ModuleList()
        # for i in range(3):
        #     self.seg_heads.append(nn.Conv2d(in_channels=256 // 2 ** i,
        #                                     out_channels=out_channels,
        #                                     kernel_size=(1, 1)))
        # self.seg_heads.append(self.seg_heads[-1])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        scale_features = [x1, x2, x3, x4, x5]

        n_stages = len(self.skeleton)

        x_skeleton, x_boundary = scale_features[-1], scale_features[-1]

        for i in range(n_stages-1, -1, -1):
            skip = scale_features[i]
            x_skeleton = self.skeleton[n_stages - (i+1)](x_skeleton, skip)
            x_boundary = self.boundary[n_stages - (i+1)](x_boundary, skip)

        x_skeleton_mask = self.ske_mask_conv(x_skeleton)
        x_boundary_mask = self.bound_mask_conv(x_boundary)

        x_fusion = torch.cat([x_skeleton, x_boundary], dim=1)
        # Segmentation-Head
        x_seg_mask = self.act(self.seg_mask_conv(self.fusion_conv(x_fusion)))

        if self.is_deepsup:
            pass
        else:
            return self.act(x_skeleton_mask), self.act(x_boundary_mask), x_seg_mask


if __name__ == '__main__':
    test = torch.ones((32, 3, 256, 256))
    model = Generator(3, 1)
    # print(model)
    out = model(test)
    print(out[0].shape, out[1].shape, out[2].shape)


