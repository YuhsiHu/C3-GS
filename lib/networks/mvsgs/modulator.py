import torch
import torch.nn as nn
import torch.nn.functional as F


class Modulater(nn.Module):
    def __init__(self, in_channel=48, mid_channel=24, out_channel=1, mode="mul"):
        super(Modulater, self).__init__()
        self.mode = mode
        if self.mode == "mul":
            self.reduce_dim_mlp = nn.Sequential(
                nn.Linear(in_channel, mid_channel),
                nn.LayerNorm(mid_channel),
                nn.GELU(),
            )
            self.gen_weight = nn.Sequential(
                nn.Linear(mid_channel, mid_channel),
                nn.LayerNorm(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, out_channel),
                nn.Sigmoid(),
            )
        elif self.mode == "cat":
            self.extract_mask_feature = nn.Sequential(
                ResidualBlock(3, mid_channel), ResidualBlock(mid_channel, mid_channel)
            )
            self.gen_weight = nn.Sequential(
                nn.LayerNorm(mid_channel + in_channel),
                nn.Linear(mid_channel + in_channel, mid_channel),
                nn.LayerNorm(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, mid_channel),
                nn.LayerNorm(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, mid_channel),
                nn.LayerNorm(mid_channel),
                nn.GELU(),
                nn.Linear(mid_channel, out_channel),
                nn.Sigmoid(),
            )

    def forward(self, raw_gaussians_pre, raw_gaussians_now, mask):
        """
        Args:
            raw_gaussians_pre: [B, H*W, 24]
            raw_gaussians_now: [B, H*W, 24]
        Return:
            weight: [B, H*W, 1]
        """
        if self.mode == "mul":
            raw_gaussians_mul = self.reduce_dim_mlp(torch.cat([raw_gaussians_pre, raw_gaussians_now], dim=-1))
            weight = self.gen_weight(raw_gaussians_mul)
        elif self.mode == "cat":
            raw_gaussians_cat = torch.cat([raw_gaussians_pre, raw_gaussians_now, mask], dim=-1)
            weight = self.gen_weight(raw_gaussians_cat)
        else:
            raise NotImplementedError
        return weight


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=nn.InstanceNorm2d,
        stride=1,
        dilation=1,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, dilation=dilation, padding=dilation, stride=stride, bias=False
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)