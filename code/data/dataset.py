
import os

import numpy as np
import torch
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):

    mins = torch.tensor([0.0, 0.0, 0.0, 0.0]) / 10000
    maxs = torch.tensor([3000.0, 3000.0, 3000.0, 7000.0]) / 10000

    def __init__(self, root, country, region, segmentation_name=None, load_esri=False, load_esawc=False):
        
        super().__init__()

        self.root = root
        self.country = country
        self.region = region
        self.segmentation_name = segmentation_name
        self.load_esri = load_esri
        self.load_esawc = load_esawc

        self.image_path = os.path.join(self.root, "Images", "Sentinel-2", "2018", self.country)
        self.lucas_path = os.path.join(self.root, "Land Cover", "LUCAS", "2018", self.country)
        self.segmentation_path = os.path.join(self.root, "Segmentations", self.segmentation_name, "2018", self.country) if self.segmentation_name not in [None, "Grid"] else None
        self.esri_path = os.path.join(self.root, "Land Cover", "ESRI", "2018", self.country) if load_esri else None
        self.esawc_path = os.path.join(self.root, "Land Cover", "ESAWC", "2020", self.country) if load_esawc else None

        self.filenames = [filename for filename in os.listdir(self.lucas_path) if (self.region in filename and filename.endswith(".pt"))]

    def __len__(self):

        return len(self.filenames)
    
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        img = torch.load(os.path.join(self.image_path, filename))
        img = normalize(img, self.mins, self.maxs)
        lucas = torch.load(os.path.join(self.lucas_path, filename))
        lc = lucas["lc"].long()
        idx = lucas["label_idx"]
        if self.segmentation_name is not None:
            if self.segmentation_name == "Grid":
                seg = torch.arange(img.shape[1] * img.shape[2], dtype=torch.long).reshape(img.shape[1], img.shape[2])
            else:
                seg = torch.load(os.path.join(self.segmentation_path, filename)).long()
        else:
            seg = torch.empty(0)
        esri = torch.load(os.path.join(self.esri_path, filename)).long() if self.load_esri else torch.empty(0)
        esawc = torch.load(os.path.join(self.esawc_path, filename)).long() if self.load_esawc else torch.empty(0)
            
        return img, lc, idx, seg, esri, esawc, filename
    
def normalize(img, mins, maxs):

    return (img - mins[:, None, None]) / (maxs[:, None, None] - mins[:, None, None])
    
def match_dense_to_sparse_labels(lc_dense, lc_sparse, idx_sparse, n_out, tolerance=0):

    lc_dense_at_idx = []
    for i in range(lc_dense.shape[0]):
        r, c = idx_sparse[i,0], idx_sparse[i,1]
        lc_dense_at_idx_i = lc_dense[i, :, r-tolerance:r+tolerance+1, c-tolerance:c+tolerance+1].flatten(start_dim=1).squeeze()
        if tolerance > 0:
            lc_dense_at_idx_i = lc_dense_at_idx_i[:, torch.argmax(lc_dense_at_idx_i[:, lc_sparse[i]])]
        lc_dense_at_idx.append(lc_dense_at_idx_i)
    lc_dense_at_idx = torch.stack(lc_dense_at_idx)
    lc_sparse = F.one_hot(lc_sparse, num_classes=n_out).float()

    return lc_dense_at_idx, lc_sparse

def remap_esawc(esawc):

    esawc_map = {
        0: 6, # undefined is set to water
        10: 2,
        20: 3,
        30: 4,
        40: 1,
        50: 0,
        60: 5,
        70: 6,
        80: 6,
        90: 7,
        95: 7,
        100: 5
    }

    esawc_harm = torch.zeros_like(esawc, dtype=torch.long)
    for old_val, new_val in esawc_map.items():
        esawc_harm[esawc == old_val] = new_val

    return esawc_harm

def remap_esri(esri):

    esri_map = {
        0: 5, # undefined is set to water
        1: 5,
        2: 2,
        3: 6,
        4: 1,
        5: 0,
        6: 4,
        7: 5,
        8: 5, # cloud cover is set to water
        9: 3
    }

    esri_harm = torch.zeros_like(esri, dtype=torch.long)
    for old_val, new_val in esri_map.items():
        esri_harm[esri == old_val] = new_val

    return esri_harm

def harmonize_lucas_esawc(lucas, esawc):

    lucas_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7
    }

    esawc_map = {
        0: 6, # undefined is set to water
        10: 2,
        20: 3,
        30: 4,
        40: 1,
        50: 0,
        60: 5,
        70: 6,
        80: 6,
        90: 7,
        95: 7,
        100: 5
    }
    
    lucas_harm = -1 * torch.ones_like(lucas, dtype=torch.long)
    for old_val, new_val in lucas_map.items():
        lucas_harm[lucas == old_val] = new_val
    
    esawc_harm = -1 * torch.ones_like(esawc, dtype=torch.long)
    for old_val, new_val in esawc_map.items():
        esawc_harm[esawc == old_val] = new_val

    return lucas_harm, esawc_harm

def harmonize_lucas_esri(lucas, esri):

    lucas_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 4,
        6: 5,
        7: 6
    }

    esri_map = {
        0: 5, # undefined is set to water
        1: 5,
        2: 2,
        3: 6,
        4: 1,
        5: 0,
        6: 4,
        7: 5,
        8: 5, # cloud cover is set to water
        9: 3
    }

    lucas_harm = -1 * torch.ones_like(lucas, dtype=torch.long)
    for old_val, new_val in lucas_map.items():
        lucas_harm[lucas == old_val] = new_val

    esri_harm = -1 * torch.ones_like(esri, dtype=torch.long)
    for old_val, new_val in esri_map.items():
        esri_harm[esri == old_val] = new_val

    return lucas_harm, esri_harm

def lc_to_img(lc):

    lc = torch.argmax(lc, dim=0) # If there is a problem, try commenting this

    colormap = [
        (0, np.array([192, 57, 43]) / 255), # Built-up
        (1, np.array([244, 208, 63]) / 255), # Cropland
        (2, np.array([11, 83, 69]) / 255), # Trees
        (3, np.array([153, 102, 51]) / 255), # Shrubland
        (4, np.array([121, 193, 113]) / 255), # Grassland
        (5, np.array([131, 145, 146]) / 255), # Bare / sparse vegetation
        (6, np.array([33, 97, 140]) / 255), # Permanent water bodies
        (7, np.array([174, 214, 241]) / 255), # Herbaceous wetland
    ]

    lc_vis = np.zeros((lc.shape[0], lc.shape[1], 3))
    for i in range(len(colormap)):
        lc_vis[lc.numpy() == colormap[i][0]] = colormap[i][1]
    
    return lc_vis