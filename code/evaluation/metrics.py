import numpy as np
from skimage.measure import label
import torch
import torch.nn.functional as F
from torchmetrics import Metric

class Entropy(Metric):

    def __init__(self, num_classes):

        super().__init__()
        self.num_classes = num_classes
        self.entropy_sum = 0
        self.n_samples = 0

    def update(self, lc_hat):

        lc_hat = F.one_hot(lc_hat, self.num_classes)
        lc_hat_mean = torch.mean(lc_hat.float(), dim=(1,2))
        entropy = - torch.sum(lc_hat_mean[lc_hat_mean > 0] * torch.log2(lc_hat_mean[lc_hat_mean > 0]))
        self.entropy_sum += entropy
        self.n_samples += lc_hat.shape[0]

    def compute(self):

        return self.entropy_sum / self.n_samples
    
    def reset(self):

        self.entropy_sum = 0
        self.n_samples = 0

class EdgeDensity(Metric):

    def __init__(self, num_classes, class_wise=False):

        super().__init__()
        self.num_classes = num_classes
        self.class_wise = class_wise
        self.n_edges = torch.zeros(num_classes, dtype=torch.long)
        self.n_pixels = torch.zeros(num_classes, dtype=torch.long)

    def update(self, lc_hat):

        self.n_pixels += torch.bincount(lc_hat.view(-1).long(), minlength=self.num_classes)

        # Horizontal edges
        h_edges_mask = lc_hat[:, 1:, :] != lc_hat[:, :-1, :]
        h_top = lc_hat[:, :-1, :][h_edges_mask]
        h_bottom = lc_hat[:, 1:, :][h_edges_mask]

        # Vertical edges
        v_edges_mask = lc_hat[:, :, 1:] != lc_hat[:, :, :-1]
        v_left = lc_hat[:, :, :-1][v_edges_mask]
        v_right = lc_hat[:, :, 1:][v_edges_mask]

        all_edge_pixels = torch.cat([h_top, h_bottom, v_left, v_right])
        self.n_edges += torch.bincount(all_edge_pixels.long(), minlength=self.num_classes)

    def compute(self):

        if self.class_wise:
            edge_density = self.n_edges.float() / (self.n_pixels.float() / 54 ** 2)
        else:
            # self.n_edges.float().sum() needs to be divided by 2 becuase each edge is counted twice in a class-wise setting
            edge_density = (self.n_edges.float().sum() / 2) / (self.n_pixels.float().sum() / 54 ** 2)
        return edge_density
    
    def reset(self):

        self.n_edges = torch.zeros(self.num_classes, dtype=torch.long)
        self.n_pixels = torch.zeros(self.num_classes, dtype=torch.long)

class PatchDensity(Metric):

    def __init__(self, num_classes=None, class_wise=False):

        super().__init__()
        self.num_classes = num_classes
        self.class_wise = class_wise
        self.n_patches = torch.zeros(num_classes, dtype=torch.long)
        self.n_pixels = torch.zeros(num_classes, dtype=torch.long)

    def update(self, lc_hat):
        
        self.n_pixels += torch.bincount(lc_hat.view(-1).long(), minlength=self.num_classes)
        lc_hat = lc_hat.numpy()
        for lc_hat_i in lc_hat:
            for cls in np.unique(lc_hat_i):
                binary_mask = (lc_hat_i == cls)
                _, count = label(binary_mask, return_num=True, connectivity=1)
                self.n_patches[int(cls)] += count

    def compute(self):

        if self.class_wise:
            patch_density = self.n_patches.float() / (self.n_pixels.float() / 54 ** 2)
        else:
            patch_density = self.n_patches.float().sum() / (self.n_pixels.float().sum() / 54 ** 2)
        return patch_density
    
    def reset(self):

        self.n_patches = torch.zeros(self.num_classes, dtype=torch.long)
        self.n_pixels = torch.zeros(self.num_classes, dtype=torch.long)