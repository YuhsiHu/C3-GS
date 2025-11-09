import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class OccLoss(torch.nn.Module):
    """
    vgg perceptual loss term for mvsgs
    """
    def __init__(self):
        super(OccLoss, self).__init__()


    def forward(self, rgb, density, reg_range=10, wb_prior=True, wb_range=20):
        '''
        Computes the occulusion regularization loss.

        Args:
            rgb (jnp.array): The RGB rays/images.
            density (jnp.array): The current density map estimate.
            reg_range (int): The number of initial intervals to include in the regularization mask.
            wb_prior (bool): If True, a prior based on the assumption of white or black backgrounds is used.
            wb_range (int): The range of RGB values considered to be a white or black background.

        Returns:
            float: The mean occlusion loss within the specified regularization range and white/black background region.
        '''
        # Compute the mean RGB value over the last dimension
        rgb_mean = rgb.mean(dim = -1)
        # Compute a mask for the white/black background region if using a prior
        if wb_prior:
            white_mask = torch.where(rgb_mean > 0.99, 1, 0) # A naive way to locate white background
            black_mask = torch.where(rgb_mean < 0.01, 1, 0) # A naive way to locate black background
            rgb_mask = (white_mask + black_mask) # White or black background
            rgb_mask[:, wb_range:] = 0 # White or black background range
        else:
            rgb_mask = torch.zeros_like(rgb_mean)
    
        # Create a mask for the general regularization region
        # It can be implemented as a one-line-code.
        if reg_range > 0:
            rgb_mask[:, :reg_range] = 1 # Penalize the points in reg_range close to the camera
    
        # Compute the density-weighted loss within the regularization and white/black background mask
        density = density.squeeze(-1)
        return torch.mean(density * rgb_mask)