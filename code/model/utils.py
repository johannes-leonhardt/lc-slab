import torch
import torch.nn.functional as F
from torch_scatter import scatter

from model.input_constructors import ImageConstructor, GraphConstructor

def reduce_lc_to_seg(lc, seg):

    lc_per_seg = []
    for i in range(lc.shape[0]):
        lc_i_flat = torch.flatten(lc[i].permute((1,2,0)), end_dim=1)
        seg_i_flat = seg[i].flatten()
        lc_per_seg.append(scatter(lc_i_flat, seg_i_flat, dim=0, reduce="mean"))
    lc_per_seg = torch.cat(lc_per_seg, dim=0)

    return lc_per_seg

def reduce_lc_to_seg_majority(lc, seg):

    lc_per_seg = []
    for i in range(lc.shape[0]):
        lc_i_flat = torch.flatten(lc[i].permute((1,2,0)), end_dim=1)
        seg_i_flat = seg[i].flatten()
        lc_i_flat = F.one_hot(torch.argmax(lc_i_flat, dim=1), num_classes=8)
        lc_i_per_seg = scatter(lc_i_flat, seg_i_flat, dim=0, reduce="sum")
        lc_i_per_seg = F.one_hot(torch.argmax(lc_i_per_seg, dim=1), num_classes=8)
        lc_per_seg.append(lc_i_per_seg)
    lc_per_seg = torch.cat(lc_per_seg, dim=0)

    return lc_per_seg

def migrate_model_compatibility(model):

    # Ensures compatability of models trained using older versions of the code with the most recent version
    # (some keywords, attributs, variables were modified during refactoring)

    if isinstance(model.input_constructor, ImageConstructor):
        if not hasattr(model.input_constructor, "kwargs"):
            setattr(model.input_constructor, "kwargs", {})
    
    if isinstance(model.input_constructor, GraphConstructor):
        if not hasattr(model.input_constructor, "node_kwargs"):
            setattr(model.input_constructor, "node_kwargs", {})
        if hasattr(model.input_constructor, "node_feature_list"):
            setattr(model.input_constructor, "node_method", model.input_constructor.node_feature_list)
            delattr(model.input_constructor, "node_feature_list")
        if "intensity_mean" in model.input_constructor.node_method:
            idx = model.input_constructor.node_method.index("intensity_mean")
            model.input_constructor.node_method[idx] = "mean"
        if "intensity_variability" in model.input_constructor.node_method:
            idx = model.input_constructor.node_method.index("intensity_variability")
            model.input_constructor.node_method[idx] = "variability"

    return model
