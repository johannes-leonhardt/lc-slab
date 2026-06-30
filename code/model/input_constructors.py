import os

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_scatter.composite import scatter_std
from torch_geometric.data import Data, Batch
from torch_geometric.utils import grid

from model.networks_px import *

class ImageConstructor(nn.Module):

    def __init__(self, kwargs={}):

        super().__init__()
        self.kwargs = kwargs
        if "feature_extractor_path" in kwargs:
            from model.sparse_lc_classifier import SparseLCClassifier
            feature_extractor_path = kwargs["feature_extractor_path"]
            self.feature_extractor = SparseLCClassifier.load_from_checkpoint(os.path.join(feature_extractor_path, "0.ckpt")).network
            self.feature_extractor.eval()

    @torch.no_grad()
    def forward(self, batch):

        im, seg = batch[0], batch[3]
        if "feature_extractor_path" in self.kwargs:
            _, im = self.feature_extractor([im, None])

        return im, seg

class GraphConstructor(nn.Module):

    def __init__(self, node_method, edge_method, node_kwargs={}, edge_kwargs={}):

        super().__init__()
        self.node_method = node_method # List of ["mean", "variability", "geometry"]
        self.node_kwargs = node_kwargs
        if "feature_extractor_path" in node_kwargs:
            from model.sparse_lc_classifier import SparseLCClassifier
            feature_extractor_path = node_kwargs["feature_extractor_path"]
            self.feature_extractor = SparseLCClassifier.load_from_checkpoint(os.path.join(feature_extractor_path, "0.ckpt"), weights_only=False).network
            self.feature_extractor.eval()
        
        self.edge_method = edge_method # One of ["None", "RAG", "Radius", kNN"]
        self.edge_kwargs = edge_kwargs
        if edge_method == "RAG":
            init_edge_index, init_pos = grid(64, 64)
            self.register_buffer("init_edge_index", init_edge_index)
            self.register_buffer("init_pos", init_pos)
        else:
            _, init_pos = grid(64, 64)
            self.register_buffer("init_pos", init_pos)

    @torch.no_grad()
    def forward(self, batch): # Data must be an image batch

        im, seg = batch[0], batch[3]

        # Nodes
        if "feature_extractor_path" in self.node_kwargs:
            _, im = self.feature_extractor([im, None])
        data_list = [self.construct_nodes(im[i], seg[i]) for i in range(im.shape[0])]
        
        # Edges
        if self.edge_method == "RAG":
            data_list = [self.construct_edges_rag(data_list[i], seg[i]) for i in range(im.shape[0])]
        batch = Batch.from_data_list(data_list)
        if self.edge_method == "None":
            pass

        return batch
    
    @torch.no_grad()
    def construct_nodes(self, img, seg): # Input must be an image sample

        x = []
        h, w = seg.shape[0], seg.shape[1]
        device = img.device
        img_flat = torch.flatten(img.permute((1,2,0)), end_dim=1)
        seg_flat = seg.flatten()
        pos = scatter(self.init_pos, seg_flat, dim=0, reduce="mean")
        if "mean" in self.node_method:
            x.append(scatter(img_flat, seg_flat, dim=0, reduce="mean"))
        if "variability" in self.node_method:
            x.append(scatter(img_flat, seg_flat, dim=0, reduce="min"))
            x.append(scatter(img_flat, seg_flat, dim=0, reduce="max"))
            x.append(scatter_std(img_flat, seg_flat, dim=0))
        if "geometry" in self.node_method:
            area = (1 / (h * w)) * scatter(torch.ones(h * w, 1).to(device), seg_flat, dim=0)
            x.append(area)
            reduced_pos = self.init_pos - pos[seg_flat]
            dist_to_center = (1 / max(h, w)) * torch.sqrt(torch.sum(reduced_pos ** 2, dim=1, keepdim=True))
            x.append(scatter(dist_to_center, seg_flat, dim=0, reduce="mean") / torch.sqrt(area))
            x.append(scatter_std(dist_to_center, seg_flat, dim=0) / torch.sqrt(area))
        x = torch.concatenate(x, dim=1)
        
        return Data(x=x, pos=pos, img=img.unsqueeze(0), seg=seg.unsqueeze(0))
    
    @torch.no_grad()
    def construct_edges_rag(self, graph, seg): # Input must be a graph sample
        
        edge_index = torch.unique(seg.flatten()[self.init_edge_index], dim=1)
        graph.edge_index = edge_index

        return graph