import sys, os
import torch
import torch.nn as nn
from torchvision import transforms
from lib.config import cfg
from lib.train.losses.vgg_perceptual_loss import VGGPerceptualLoss
from lib.train.losses.ssim_loss import SSIM
from lib.train.losses.occ_loss import OccLoss

class NetworkWrapper(nn.Module):
    """
    final loss function for mvsgs
    """
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        self.occ_loss = OccLoss()

    def forward(self, batch):
        # get the output from the network
        output = self.net(batch)
        scalar_stats = {}
        image_stats = {}
        inp_0, tar_0 = None, None
        loss = 0
        for i in range(cfg.mvsgs.cas_config.num):
            if not cfg.mvsgs.cas_config.render_if[i]:
                continue
            # color loss
            color_loss = self.color_crit(batch[f'rgb_{i}'], output[f'rgb_level{i}'])
            scalar_stats.update({f'color_mse_{i}': color_loss})
            loss += 1.0 * color_loss * cfg.mvsgs.cas_config.loss_weight[i]
            psnr = -10. * torch.log(color_loss) / torch.log(torch.Tensor([10.]).to(color_loss.device))
            scalar_stats.update({f'psnr_{i}': psnr})
            #### occloss ####
            occ_loss = self.occ_loss(rgb = batch[f'rgb_{i}'], density = output[f'sigma_level{i}'])
            loss += 1.0 * occ_loss * cfg.mvsgs.cas_config.loss_weight[i]
            scalar_stats.update({f'occ_loss_{i}': occ_loss.detach()})
            # ssim loss and perceptual loss
            num_patchs = cfg.mvsgs.cas_config.num_patchs[i] # 0 if not patch
            if cfg.mvsgs.cas_config.train_img[i]:
                # this way
                render_scale = cfg.mvsgs.cas_config.render_scale[i]
                B, S, C, H, W = batch['src_inps'].shape
                H, W = int(H * render_scale), int(W * render_scale)
                # rendered image from network
                inp = output[f'rgb_level{i}'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
                # target image from dataloader
                tar = batch[f'rgb_{i}'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
                inp_0 = inp
                tar_0 = tar
                perceptual_loss = self.perceptual_loss(inp, tar)
                loss += 0.05 * perceptual_loss * cfg.mvsgs.cas_config.loss_weight[i]
                scalar_stats.update({f'perceptual_loss_{i}': perceptual_loss.detach()})
                ssim = SSIM(window_size = 7)
                ssim_loss = 1 - ssim(inp, tar)
                loss += 0.1 * ssim_loss * cfg.mvsgs.cas_config.loss_weight[i]
                scalar_stats.update({f'ssim_loss_{i}': ssim_loss.detach()})
                
            elif num_patchs > 0:
                patch_size = cfg.mvsgs.cas_config.patch_size[i]
                num_rays = cfg.mvsgs.cas_config.num_rays[i]
                patch_rays = int(patch_size ** 2)
                inp = torch.empty((0, 3, patch_size, patch_size)).to(self.device)
                tar = torch.empty((0, 3, patch_size, patch_size)).to(self.device)
                for j in range(num_patchs):
                    inp = torch.cat([inp, output[f'rgb_level{i}'][:, num_rays+j*patch_rays:num_rays+(j+1)*patch_rays, :].reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)])
                    tar = torch.cat([tar, batch[f'rgb_{i}'][:, num_rays+j*patch_rays:num_rays+(j+1)*patch_rays, :].reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)])
                perceptual_loss = self.perceptual_loss(inp, tar)

                loss += 0.01 * perceptual_loss * cfg.mvsgs.cas_config.loss_weight[i]
                scalar_stats.update({f'perceptual_loss_{i}': perceptual_loss.detach()})

        scalar_stats.update({'loss': loss})
        image_stats.update({'rgb_0': tar_0})
        image_stats.update({'rendered_level0': inp_0})
        # gs and vr
        image_stats.update({"rgb_gs": output['render_novel_i_0']})
        image_stats.update({"rgb_vr": output['rgb_vr']})

        return output, loss, scalar_stats, image_stats

