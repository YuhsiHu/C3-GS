import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np


class MvsDepthLoss(torch.nn.Module):
    def __init__(self):
        super(MvsDepthLoss, self).__init__()


    def forward(self, pred_depth, gt_depth, mask):
        '''
        pred_depth: render depth [B, 1, H, W]
        gt_depth: mvs_depth from pretrained model [B, H, W]
        '''
        print(pred_depth.shape, gt_depth.shape, mask.shape)
        
        return torch.mean(torch.abs((pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]))