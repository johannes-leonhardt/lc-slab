import os

import geopandas as gpd
from skimage.segmentation import felzenszwalb, slic
from skimage.exposure import adjust_gamma
from tqdm import tqdm
import torch

from data.dataset import Dataset

class Segmentor():

    '''
    Featured options:
    - SLIC
    - Felzenszwalb
    '''

    def __init__(self, segmentation_method, segmentation_kwargs={}, normalization_method=None, normalization_kwargs={}):

        # Segmentation
        self.segmentation_method = segmentation_method
        if segmentation_method == "SLIC":
            self.segmentation_func = slic
        elif segmentation_method == "Felzenszwalb":
            self.segmentation_func = felzenszwalb
        self.segmentation_kwargs = segmentation_kwargs

        # Normalization
        self.normalization_method = normalization_method
        if normalization_method == "Gamma":
            self.normalization_func = adjust_gamma
        self.normalization_kwargs = normalization_kwargs

    def segment_sample(self, im):

        im = im.permute((1,2,0)).numpy()
        if self.normalization_method is not None:
            im_norm = self.normalization_func(im, **self.normalization_kwargs)
        seg = self.segmentation_func(im_norm, **self.segmentation_kwargs)
        seg = torch.tensor(seg, dtype=torch.int16)
        
        return seg
    
    def segment_all_and_save(self, data_root):
        
        regions = gpd.read_file(os.path.join(data_root, "lucas_regions.gpkg"))
        segmentation_name = self.segmentation_method
        for key, val in self.segmentation_kwargs.items():
            segmentation_name += f"_{key}_{val}"
        seg_dir = os.path.join(data_root, "Segmentations", segmentation_name, "2018")

        for i in tqdm(regions.index, desc=segmentation_name):

            country = regions.country.loc[i]
            region = regions.region.loc[i]
            ds = Dataset(data_root, country, region)
            seg_dir_i = os.path.join(seg_dir, country)
            if not os.path.exists(seg_dir_i):
                os.makedirs(seg_dir_i)
            for img, _, _, _, _, _, filename in tqdm(ds, desc=f"Segmenting {country}, {region}..."):
                seg = self.segment_sample(img)
                torch.save(seg, os.path.join(seg_dir_i, filename))

        return
