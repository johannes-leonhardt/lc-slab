import os
gpus = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import warnings
from datetime import datetime
import yaml

import geopandas as gpd
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy

from data.dataset import Dataset
from model.sparse_lc_classifier import *
from model.input_constructors import *
from model.networks_px import *
from model.networks_obj import *
from model.postprocessing_functions import *
from model.loss_functions import *

## Preliminaries
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
n_gpus = len(gpus.split(','))

## Configuration
experiment_name = "Test/GNN_Pretrained"

config = {

    # Object definition - Specify any segmentation or "Grid" for input-level obj. aggr. or "Grid" for output-level obj. aggr.
    "segmentation_name": "Felzenszwalb_scale_1_sigma_0_min_size_40",

    # Input constructor - Specify "GraphConstructor" for input-level obj. aggr. or "ImageConstructor" for output-level obj. aggr.
    "input_constructor": GraphConstructor, 
    "input_constructor_kwargs": {

    #     # For ImageConstructor
    #     "kwargs": {
    #         # "feature_extractor_path": os.path.join(
    #         #     "..", 
    #         #     "out", 
    #         #     "Pretrained", 
    #         #     "UPerNet_Resnet152", 
    #         #     "2025-08-01_00-02-33_ImageConstructor_UPerNet_SMP_DenseCrossEntropy", 
    #         #     "model_checkpoints"
    #         # )
    #     },

        # For GraphConstructor
        "node_method": ["mean"], # in ["mean", "variability", "geometry"]
        "node_kwargs": {
            "feature_extractor_path": os.path.join(
                "..", 
                "out", 
                "Pretrained", 
                "UPerNet_Resnet152", 
                "2026-06-30_12-52-58_ImageConstructor_UPerNet_SMP_DenseCrossEntropy", 
                "model_checkpoints"
            )
        },
        "edge_method": "RAG",
        "edge_kwargs": {},

    },

    # Classifier
    "network": GNN,
    "network_kwargs": {
        "n_in": 64, # No.feats for intensity, No.feats * 3 for variability, 3 for geometry
        "n_out": 8,
        "n_layers": 3, # 3 for MLP, CNN, GNN and GraphUNet; 5 for CNN-encoders
        # "encoder": "mitb3",
        # "upscale": False, # 64 x 64 -> 224 x 224
        "n_hidden": 256, # 256 for MLP, CNN and GNN; 64 for GraphUNet
        "conv": pygnn.TransformerConv,
        "conv_kwargs": {
            # "aggr": "mean", # for SAGE
            "heads": 4, # for GAT and GT
            "concat": False, # for GAT and GT
        }
    },

    # Loss function
    "loss_function": PartialCrossEntropy,
    "loss_function_kwargs": {},

    # Postprocessing (change during testing/inference for output-level obj. aggr.)
    "postprocessing_function": NoFilter,
    "postprocessing_function_kwargs": {},

    # Other settings
    "n_runs": 1,
    "ds_fraction": 1,
    "max_epochs": 20,
    "batch_size": 64,
    "gpus": gpus
}

## Dataset preparation
now = datetime.now()
data_root = os.path.join("..", "data")
regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
train_regions = regions[regions.split == "train"]
val_regions = regions[regions.split == "val"]
train_ds, val_ds = [], []
for country, region in zip(train_regions.country, train_regions.region):
    train_ds.append(Dataset(data_root, country, region, config["segmentation_name"]))
for country, region in zip(val_regions.country, val_regions.region):
    val_ds.append(Dataset(data_root, country, region, config["segmentation_name"]))
train_ds, val_ds = ConcatDataset(train_ds), ConcatDataset(val_ds)

# Select random subset
if config["ds_fraction"] < 1:
    generator = torch.Generator().manual_seed(0)
    perm_train, perm_val = torch.randperm(len(train_ds), generator=generator), torch.randperm(len(val_ds), generator=generator)
    train_ds, val_ds = Subset(train_ds, perm_train[:int(config["ds_fraction"] * len(train_ds))]), Subset(val_ds, perm_val[:int(config["ds_fraction"] * len(val_ds))])

## Data loader preparation
local_batch_size = config["batch_size"] // n_gpus
train_loader = DataLoader(train_ds, shuffle=True, batch_size=local_batch_size, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=local_batch_size, num_workers=0, drop_last=True)

## Prepare directories for saving checkpoints and results
model_name = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{config['input_constructor'].__name__}_{config['network'].__name__}_{config['loss_function'].__name__}"
output_path = os.path.join("..", "out", experiment_name, model_name)
checkpoint_path = os.path.join(output_path, "model_checkpoints")
if os.getenv("LOCAL_RANK", '0') == '0':
    os.makedirs(checkpoint_path)
    with open(os.path.join(output_path, "config.yml"), "w") as outfile: 
        yaml.dump(config, outfile, default_flow_style=False)

## Training
for i in range(config["n_runs"]):

    # Initialize classifier
    model = SparseLCClassifier(
        input_constructor = config["input_constructor"](**config["input_constructor_kwargs"]),
        network = config["network"](**config["network_kwargs"]),
        loss_function = config["loss_function"](**config["loss_function_kwargs"]),
        postprocessing_function = config["postprocessing_function"](**config["postprocessing_function_kwargs"]),
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        devices=n_gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True), # Some settings require "True"
        precision="16-mixed",
        logger=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename=f"{i}",
                save_top_k=1,
                monitor="val_acc", 
                mode="max"
            )
        ]
    )

    # Train
    trainer.fit(model, train_loader, val_loader)