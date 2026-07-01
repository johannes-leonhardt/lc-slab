import os
gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import geopandas as gpd
import torch
from pytorch_lightning.utilities import model_summary
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm

from data.dataset import Dataset, match_dense_to_sparse_labels
from model.sparse_lc_classifier import SparseLCClassifier
from model.postprocessing_functions import *
from model.utils import migrate_model_compatibility
from evaluation.metrics import Entropy, EdgeDensity, PatchDensity

## Model settings
experiment_name = "02c_ObjLevelNetworks_SLIC/MLP"
experiment_dir = os.path.join("..", "out", experiment_name)
# Automatically read all from experiment folder ...
model_names = sorted([name for name in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, name))])
# ... or specify manually
# model_names = ["2025-08-22_10-05-53_GraphConstructor_GNN_PartialCrossEntropy"]
batch_size = 128
class_wise_metrics = False

# Specify Segmentation as postprocessing function for output-level aggregation
postprocessing_config = None
# postprocessing_config = {
#     "postprocessing_function": Segmentation,
#     "postprocessing_function_kwargs": {}
# }

for model_name in model_names:

    ## Get some settings from config file
    with open(os.path.join(experiment_dir, model_name, "config.yml")) as config_file:
        lines = config_file.readlines()
        for line in lines:
            if "n_out" in line:
                n_out = int(line.split()[-1])
            if "n_runs" in line:
                n_runs = int(line.split()[-1])
            if ("segmentation_name" in line) & ("null" not in line):
                segmentation_name = line.split()[-1]

    print(segmentation_name)

    ## Data handling
    data_root = os.path.join(os.path.sep, "data2", "jleonhardt", "LC-SLab")
    regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
    regions = regions[regions.split == "test"]

    output_path = os.path.join("..", "output_depr", experiment_name, model_name)
    print(output_path)
    checkpoint_path = os.path.join(output_path, "model_checkpoints")
    checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith(".ckpt")]

    # Metrics
    acc_0 = Accuracy(task="multiclass", num_classes=n_out, average="micro")
    f1_0 = F1Score(task="multiclass", num_classes=n_out, average="macro")
    acc_1 = Accuracy(task="multiclass", num_classes=n_out, average="micro")
    f1_1 = F1Score(task="multiclass", num_classes=n_out, average="macro")
    entropy = Entropy(num_classes=n_out)
    edge_dens = EdgeDensity(num_classes=n_out)
    patch_dens = PatchDensity(num_classes=n_out)
    if class_wise_metrics:
        acc_0_class = Accuracy(task="multiclass", num_classes=n_out, average="none")
        f1_0_class = F1Score(task="multiclass", num_classes=n_out, average="none")
        acc_1_class = Accuracy(task="multiclass", num_classes=n_out, average="none")
        f1_1_class = F1Score(task="multiclass", num_classes=n_out, average="none")
        edge_dens_class = EdgeDensity(num_classes=n_out, class_wise=True)
        patch_dens_class = PatchDensity(num_classes=n_out, class_wise=True)

    # Testing
    for i in range(n_runs):
        
        checkpoint = f"{i}.ckpt"
        print(f"Testing model checkpoint {checkpoint}...")

        # Load model from checkpoint
        model = SparseLCClassifier.load_from_checkpoint(os.path.join(checkpoint_path, checkpoint), weights_only=False)
        model = migrate_model_compatibility(model)
        if postprocessing_config is not None:
            model.postprocessing_function = postprocessing_config["postprocessing_function"](**postprocessing_config["postprocessing_function_kwargs"])
        if i == 0:
            print(model_summary.summarize(model))

        # Collect results by applying model to test dataset
        for ii in tqdm(regions.index):

            # Apply model to dataset of region
            country = regions.country.loc[ii]
            region = regions.region.loc[ii]
            ds = Dataset(data_root, country, region, segmentation_name)
            if len(ds) == 0: # There are some regions without any LUCAS points
                continue
            lc_hat, lc_lucas, idx_lucas = model.apply_to_ds(ds, batch_size=batch_size)

            # Update accuracy metrics
            lc_hat_cmp, lc_lucas_cmp = match_dense_to_sparse_labels(lc_hat, lc_lucas, idx_lucas, n_out, tolerance=0)
            acc_0.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
            f1_0.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
            if class_wise_metrics:
                acc_0_class.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
                f1_0_class.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
            lc_hat_cmp, lc_lucas_cmp = match_dense_to_sparse_labels(lc_hat, lc_lucas, idx_lucas, n_out, tolerance=1)
            acc_1.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
            f1_1.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
            if class_wise_metrics:
                acc_1_class.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))
                f1_1_class.update(torch.argmax(lc_hat_cmp, dim=1), torch.argmax(lc_lucas_cmp, dim=1))

            # Update complexity metrics
            lc_hat = lc_hat[:, :, 5:-5, 5:-5] # Ignore edges of images
            entropy.update(torch.argmax(lc_hat, dim=1))
            edge_dens.update(torch.argmax(lc_hat, dim=1))
            patch_dens.update(torch.argmax(lc_hat, dim=1))
            if class_wise_metrics:
                edge_dens_class.update(torch.argmax(lc_hat, dim=1))
                patch_dens_class.update(torch.argmax(lc_hat, dim=1))

        # Compute metrics
        acc_0_result = acc_0.compute().item()
        f1_0_result = f1_0.compute().item()
        acc_1_result = acc_1.compute().item()
        f1_1_result = f1_1.compute().item()
        entropy_result = entropy.compute().item()
        edge_dens_result = edge_dens.compute().item()
        patch_dens_result = patch_dens.compute()
        if class_wise_metrics:
            acc_0_class_result = acc_0_class.compute()
            f1_0_class_result = f1_0_class.compute()
            acc_1_class_result = acc_1_class.compute()
            f1_1_class_result = f1_1_class.compute()
            edge_dens_class_result = edge_dens_class.compute()
            patch_dens_class_result = patch_dens_class.compute()

        # Save results
        results_path = os.path.join(output_path, f"results", checkpoint.replace(".ckpt",""))
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
        if class_wise_metrics:
            with open(os.path.join(results_path, f"metrics_classwise.txt"), "a") as output:
                output.write("\nCollected metrics\n---\n")
                output.write(f"Overall accuracy (t=0): {list(acc_0_class_result)}\n")
                output.write(f"Average F1 score (t=0): {list(f1_0_class_result)}\n")
                output.write(f"Overall accuracy (t=1): {list(acc_1_class_result)}\n")
                output.write(f"Average F1 score (t=1): {list(f1_1_class_result)}\n")
                output.write(f"Average number of edges per sample: {list(edge_dens_class_result)}\n")
                output.write(f"Average number of patches per sample: {list(patch_dens_class_result)}\n")

        # Prepare testing of next checkpoint
        acc_0.reset()
        f1_0.reset()
        acc_1.reset()
        f1_1.reset()
        entropy.reset()
        edge_dens.reset()
        patch_dens.reset()
        if class_wise_metrics:
            acc_0_class.reset()
            f1_0_class.reset()
            acc_1_class.reset()
            f1_1_class.reset()
            edge_dens_class.reset()
            patch_dens_class.reset()