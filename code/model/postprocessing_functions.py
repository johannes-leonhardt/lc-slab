
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d, bilateral_blur

from model.utils import reduce_lc_to_seg

class NoFilter(nn.Module):

    def __init__(self):
        
        super().__init__()

    def forward(self, result):

        lc_hat, _ = result

        return lc_hat
    
class Segmentation(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, result):

        lc_hat, network_input = result
        _, seg = network_input
        for i in range(lc_hat.shape[0]):
            lc_hat_per_seg = reduce_lc_to_seg(lc_hat[[i]], seg[[i]])
            lc_hat[i] = lc_hat_per_seg[seg[i]].permute(2, 0, 1) # Remap graph to raster

        return lc_hat
