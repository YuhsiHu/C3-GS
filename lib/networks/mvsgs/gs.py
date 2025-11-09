import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from .utils import *
from .feature_net import Unet
from .cost_reg_net import SigCostRegNet
from .transformer import MultiHeadAttention
from .modulator import Modulater


class GS(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        """
        """
        super(GS, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.head_dim = 24
        self.Unet = Unet(self.head_dim, 16)
        self.opacity_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )
        # self.rotation_head = nn.Sequential(
        #     nn.Linear(self.head_dim, self.head_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.head_dim, 4),
        # )
        self.scale_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, 3),
            nn.Softplus()
        )
        self.color = nn.Sequential(
            nn.Linear(feat_ch+self.head_dim+4, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, 1),
            nn.GELU())
        self.sigma = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Softplus())
        self.color_gs = nn.Sequential(
            nn.Linear(self.head_dim+3, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, 3),
            nn.Sigmoid()
        )
        ##### attention #####
        self.lr0 = nn.Sequential(nn.Linear(8+16, hid_n), nn.GELU())
        self.lrs = nn.ModuleList([nn.Sequential(nn.Linear(hid_n, self.head_dim), nn.GELU()) for i in range(1)])
        self.regnet = SigCostRegNet(hid_n)
        self.attn = MultiHeadAttention(n_head=4, d_model=hid_n, d_model_k=feat_ch+4, d_k=4, d_v=4)
        #####################
        #### erroraware ####
        self.modulater = Modulater(in_channel=2*24, mid_channel=24, out_channel=1, mode="mul")
        ####################


    def forward(self, vox_feat, img_feat_rgb_dir, z_vals, batch, size, level, prev_gaussians=None):
        """Volume rendering and estimation of Gaussian parameters
        Args:
            vox_feat: [B, H*W, 8]
            img_feat_rgb_dir: [B, H*W, S, 3+C+4], C=32 or 8
            z_vals: [B, H*W, 1]
            batch: info from dataloader
            size: (H, W)
            level: 0 or 1
        Returns:
            world_xyz: [B, H*W, 3]
            rot_out: [B, H*W, 4]
            scale_out: [B, H*W, 3]
            opacity_out: [B, H*W, 1]
            color_out: [B, H*W, 3]
            rgb_vr: [B, 3, H, W]
            sigma: [B, H*W, 1] 
            img_feat: [B, H*W, 16]
            x: [B, H*W, 24]
        """
        H, W = size
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        S, C_dim = img_feat_rgb_dir.shape[2], img_feat_rgb_dir.shape[-1]
        # aggregate source features
        img_feat = self.agg(img_feat_rgb_dir) # [B, H*W, S, 3+C+4] -> [B, H*W, 16]
        # #####
        # x = torch.cat((vox_feat, img_feat), dim=-1) # [B, H*W, 24]
        # #####
        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1) # [B, H*W, 8] [B, H*W, 16] -> [B, H*W, 24]
        ##### attention #####
        x = self.lr0(vox_img_feat) # [B, H*W, 64]
        if level == 0:
            x = x.reshape(B, H, W, -1, x.shape[-1]) # [B, H, W, 1, 64]
            x = x.permute(0, 4, 3, 1, 2) # [B, 64, 1, H, W]
            x = self.regnet(x) # [B, 64, 1, H, W]
            x = x.permute(0, 1, 3, 4, 2).flatten(2).permute(0, 2, 1) # [B, 64, 1, H, W] -> [B, 64, H, W, 1] -> [B, 64, H*W*1] -> [B, H*W, 64]
            # create qkv
            q = x.reshape(B*H*W, 1, 64) # [B*H*W, 1, 64]
            k = img_feat_rgb_dir.view(B*H*W, S, C_dim) # [B*H*W, S, C+7]
            x, _ = self.attn(q, k, k) # [H*W, B, 64]
            x = x.permute(1, 0, 2) # [B, H*W, 64]
        for i in range(len(self.lrs)):
            x = self.lrs[i](x) # [B, H*W, 24]
        #####################
        # depth 
        d = z_vals.shape[-1]
        z_vals = z_vals.reshape(B, H, W, d) # [B, H, W, 1]
        if cfg.mvsgs.cas_config.depth_inv[level]:
            z_vals = 1./torch.clamp_min(z_vals, 1e-6) # to disp
        depth = z_vals.permute(0, 3, 1, 2) # [B, 1, H, W]
        # sigma head
        sigma = self.sigma(x) # [B, H*W, 1]
        # radiance head
        x0 = x.unsqueeze(2).repeat(1, 1, S, 1) # [1, H*W, S, 24]
        x0 = x0.reshape(B, H*W, S, -1) # [B, H*W, S, 24]
        x0 = torch.cat((x0, img_feat_rgb_dir), dim=-1) # [B, H*W, S, 24+C+7]
        color_weight = F.softmax(self.color(x0), dim=-2) # [B, H*W, S, 1] view weighting for color contribution across views
        radiance = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2) # [B, H*W, 3] final rgb for points
        # volume rendering branch
        sigma = sigma.reshape(B, H*W, d) # [B, H*W, 1] density(opacity) per point
        raw2alpha = lambda raw: 1.-torch.exp(-raw)
        alpha = raw2alpha(sigma)  # [B, H*W, 1]
        T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1] # [B, H*W, 0]
        T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1) # [B, H*W, 1] transmittance computed as a cumulative product of the opacity
        weights = alpha * T # [B, H*W, 1] contribution of each point along the ray
        radiance = radiance.reshape(B, H*W, d, 3) # [B, H*W, 1, 3]
        rgb_vr = torch.sum(weights[..., None] * radiance, -2) # [B, H*W, 3]
        rgb_vr = rgb_vr.reshape(B, H, W, 3).permute(0, 3, 1, 2) # [B, 3, H, W]
        # enhance features using a UNet
        x = x.reshape(B, H*W, d, x.shape[-1])
        x = torch.sum(weights[..., None].unsqueeze(0) * x, -2)  # [N_rays, 3]
        x = x.reshape(B, H, W, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.Unet(x)
        x = x.flatten(-2).permute(0, 2, 1) # [B, H*W, 24]
        # gs branch
        # rot head
        rot_out = torch.ones((B, x.shape[1], 4)).to(x.device)
        # rot_out = self.rotation_head(x)
        rot_out = torch.nn.functional.normalize(rot_out, dim=-1) # [B, H*W, 4]    
        # scale head
        scale_out = self.scale_head(x) # [B, H*W, 3]
        # opacity head
        opacity_out = self.opacity_head(x) # [B, H*W, 1]
        #### erroraware ####
        if level != 0:
            opacity_weight = self.modulater(prev_gaussians, x, mask=None)
            opacity_out = opacity_out * opacity_weight
        ####################
        # color head
        x0 = torch.cat((x, rgb_vr.flatten(2).permute(0, 2, 1)), dim=-1)
        color_out = self.color_gs(x0) # [B, H*W, 3]
        # world_xyz
        weights = weights.reshape(B, H, W, d).permute(0, 3, 1, 2) # [B, H*W, 1] -> [B, 1, H, W]
        depth = torch.sum(weights * depth, 1) # [B, H, W]
        ext = batch['tar_ext'] # [B, 4, 4]
        ixt = batch['tar_ixt'].clone() # [B, 3, 3]
        ixt[:,:2] *= cfg.mvsgs.cas_config.render_scale[level]
        world_xyz = depth2point(depth, ext, ixt) # [B, H*W, 3]

        # return world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr
        return world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr, sigma, x


class Agg(nn.Module):
    def __init__(self, feat_ch):
        """
        aggregate image features from different views
        """
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        if cfg.mvsgs.viewdir_agg:
            self.view_fc = nn.Sequential(
                    nn.Linear(4, feat_ch),
                    nn.ReLU(),
                    )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
                nn.Linear(feat_ch*3, 32),
                nn.ReLU(),
                )

        self.agg_w_fc = nn.Sequential(
                nn.Linear(32, 1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, img_feat_rgb_dir):
        """
        Args:
            img_feat_rgb_dir: Source features from S views with shape [B, H*W, S, 3+C+4],

        Returns:
            Tensor: Aggregated feature representation [B, H*W, 16].

        Processing Steps:
            1. **View Direction Processing (optional)**: Applies a fully connected layer to the view direction features if enabled by `cfg.mvsgs.viewdir_agg`.
            2. **Variance & Average Calculation**: Computes variance and average across views and repeats them per sample.
            3. **Global Feature Aggregation**: Concatenates features and applies fully connected layers to obtain weighted features.
            4. **Softmax Weighting**: Computes attention weights for features from each view.
            5. **Final Feature Extraction**: Outputs aggregated features with 16 channels.
        """
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]
        if cfg.mvsgs.viewdir_agg:
            # this way
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:]) # [B, H*W, S, 4] -> [B, H*W, S, 3+C]
            img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat # [B, H*W, S, 3+C] the original source features (RGB + C) enhanced with the processed view direction features 
        else:
            img_feat_rgb =  img_feat_rgb_dir[..., :-4]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1) # [B, H*W, 3+C] -> [B, H*W, 1, 3+C] -> [B, H*W, S, 3+C]
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1) # [B, H*W, 3+C] -> [B, H*W, 1, 3+C] -> [B, H*W, S, 3+C]

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1) # [B, H*W, S, (3+C)*3]
        global_feat = self.global_fc(feat) # [B, H*W, S, (3+C)*3] -> [B, H*W, S, 32]
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2) # [B, H*W, S, 32] -> [B, H*W, S, 1] weight for each view across each pixel
        im_feat = (global_feat * agg_w).sum(dim=-2) # [B, H*W, S, 32] -> [B, H*W, 32] combinine the features from each view according to the computed attention weights.
        return self.fc(im_feat) # [B, H*W, 32] -> [B, H*W, 16]


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)