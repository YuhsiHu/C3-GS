import torch.nn as nn
from .utils import *
from .intra_view import IntraViewNet


class FeatureNet(nn.Module):
    """
    feature extraction in mvsnet
    """

    def __init__(self, norm_act=nn.BatchNorm2d):
        super(FeatureNet, self).__init__()
        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

        # intra view
        self.intraview1 = IntraViewNet(32, 32)
        self.intraview0 = IntraViewNet(32, 32)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)  # [B*S, 8, H, W]
        conv1 = self.conv1(conv0)  # [B*S, 16, H/2, W/2]
        conv2 = self.conv2(conv1)  # [B*S, 32, H/4, W/4]
        feat2 = self.toplayer(conv2)  # [B*S, 32, H/4, W/4]
        del conv2
        # feat1 = self._upsample_add(feat2, self.lat1(conv1)) # [B*S, 32, H/2, W/2]
        #### intra view ####
        feat1 = self._upsample_add(feat2, self.intraview1(self.lat1(conv1)))  # [B*S, 32, H/2, W/2]
        #####################
        del conv1
        # feat0 = self._upsample_add(feat1, self.lat0(conv0)) # [B*S, 32, H, W]
        #### intra view ####
        feat0 = self._upsample_add(feat1, self.intraview0(self.lat0(conv0)))  # [B*S, 32, H, W]
        #####################
        del conv0
        feat1 = self.smooth1(feat1)  # [B*S, 32, H/2, W/2]
        feat0 = self.smooth0(feat0)  # [B*S, 32, H, W]
        return feat2, feat1, feat0

class Unet(nn.Module):
    """
    enhance feature in gs module
    """

    def __init__(self, in_channels, base_channels, norm_act=nn.BatchNorm2d):
        super(Unet, self).__init__()
        self.conv0 = nn.Sequential(
            ConvBnReLU(in_channels, base_channels, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(base_channels, base_channels, 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(base_channels, base_channels * 2, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(base_channels * 2, base_channels * 2, 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(base_channels * 2, base_channels * 4, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(base_channels * 4, base_channels * 4, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(base_channels * 4, base_channels * 4, 1)
        self.lat1 = nn.Conv2d(base_channels * 2, base_channels * 4, 1)
        self.lat0 = nn.Conv2d(base_channels, base_channels * 4, 1)

        self.smooth0 = nn.Conv2d(base_channels * 4, in_channels, 3, padding=1)

        # intra view
        self.intraview1 = IntraViewNet(64, 64)
        self.intraview0 = IntraViewNet(64, 64)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)  # [B, 16, H, W]
        conv1 = self.conv1(conv0)  # [B, 32, H/2, W/2]
        conv2 = self.conv2(conv1)  # [B, 64, H/4, W/4]
        feat2 = self.toplayer(conv2)  # [B, 64, H/4, W/4]
        del conv2
        feat1 = self._upsample_add(feat2, self.intraview1(self.lat1(conv1)))
        del conv1
        feat0 = self._upsample_add(feat1, self.intraview0(self.lat0(conv0)))
        del conv0
        feat0 = self.smooth0(feat0)
        return feat0
