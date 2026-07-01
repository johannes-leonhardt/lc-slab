import os
gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import numpy as np
import geopandas as gpd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
from tqdm import tqdm

from data.dataset import Dataset, match_dense_to_sparse_labels, harmonize_lucas_esri, harmonize_lucas_esawc
from evaluation.metrics import Entropy, EdgeDensity, PatchDensity

## Settings
experiment_name = "00_OtherProducts"
product_name = "ESRI"

## Data handling
data_root = os.path.join("..", "data")
regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
regions = regions[regions.split == "test"]
if product_name == "ESRI":
    n_out = 7
elif product_name == "ESAWC":
    n_out = 8

# Metrics
acc_0 = Accuracy(task="multiclass", num_classes=n_out)
f1_0 = F1Score(task="multiclass", num_classes=n_out, average="macro")
acc_1 = Accuracy(task="multiclass", num_classes=n_out)
f1_1 = F1Score(task="multiclass", num_classes=n_out, average="macro")
entropy = Entropy(num_classes=n_out)
edge_dens = EdgeDensity(num_classes=n_out)
patch_dens = PatchDensity(num_classes=n_out)

## Testing
for i in tqdm(regions.index):
    country = regions.country.loc[i]
    region = regions.region.loc[i]
    ds = Dataset(data_root, country, region, load_esri=product_name=="ESRI", load_esawc=product_name=="ESAWC")
    dl = DataLoader(ds, batch_size=len(ds))
    if len(ds) == 0: # There are some regions without any LUCAS points
        continue
    
    # Semantically align third-party product and ground truth
    _, lc_lucas, idx_lucas, _, lc_esri, lc_esawc, _ =  next(iter(dl))
    if product_name == "ESRI":
        lc_lucas, lc_product = harmonize_lucas_esri(lc_lucas, lc_esri)
    elif product_name == "ESAWC":
        lc_lucas, lc_product = harmonize_lucas_esawc(lc_lucas, lc_esawc)
    lc_product = F.one_hot(lc_product.squeeze(), n_out).permute(0,3,1,2).float()

    # Update accuracy metrics
    lc_product_cmp, lc_lucas_cmp = match_dense_to_sparse_labels(lc_product, lc_lucas, idx_lucas, n_out, tolerance=0)
    acc_0.update(torch.argmax(lc_product_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
    f1_0.update(torch.argmax(lc_product_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
    lc_product_cmp, lc_lucas_cmp = match_dense_to_sparse_labels(lc_product, lc_lucas, idx_lucas, n_out, tolerance=1)
    acc_1.update(torch.argmax(lc_product_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
    f1_1.update(torch.argmax(lc_product_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))

    # Update complexity metrics
    lc_product = lc_product[:, :, 5:-5, 5:-5]
    entropy.update(torch.argmax(lc_product, dim=1))
    edge_dens.update(torch.argmax(lc_product, dim=1))
    patch_dens.update(torch.argmax(lc_product, dim=1))

# Compute metrics
acc_0_result = acc_0.compute().item()
f1_0_result = f1_0.compute().item()
acc_1_result = acc_1.compute().item()
f1_1_result = f1_1.compute().item()
entropy_result = entropy.compute().item()
edge_dens_result = edge_dens.compute().item()
patch_dens_result = patch_dens.compute()

# Save results
results_path = os.path.join("..", "out", "00_OtherProducts", product_name)
try:
    os.makedirs(results_path)
except FileExistsError:
    pass
with open(os.path.join(results_path, f"metrics.txt"), "a") as output:
    output.write("\nCollected metrics\n---\n")
    output.write(f"Overall accuracy (t=0): {acc_0_result}\n")
    output.write(f"Average F1 score (t=0): {f1_0_result}\n")
    output.write(f"Overall accuracy (t=1): {acc_1_result}\n")
    output.write(f"Average F1 score (t=1): {f1_1_result}\n")
    output.write(f"Average sample-wise entropy: {entropy_result}\n")
    output.write(f"Average number of edges per sample: {edge_dens_result}\n")
    output.write(f"Average number of patches per sample: {patch_dens_result}\n")