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
experiment_name = "Pretrained/UPerNet_Resnet152"
config = {

    # Input constructor
    "input_constructor": ImageConstructor,
    "input_constructor_kwargs": {},

    # Classifier
    "network": UPerNet_SMP,
    "network_kwargs": {
        "n_in": 4,
        "n_out": 8,
        "n_layers": 5,
        "encoder": "resnet152",
    },

    # Loss function
    "loss_function": DenseCrossEntropy,
    "loss_function_kwargs": {},

    # Postprocessing
    "postprocessing_function": NoFilter,
    "postprocessing_function_kwargs": {},

    # Other settings
    "n_runs": 1,
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
    train_ds.append(Dataset(data_root, country, region, load_esawc=True))
for country, region in zip(val_regions.country, val_regions.region):
    val_ds.append(Dataset(data_root, country, region, load_esawc=True))
train_ds, val_ds = ConcatDataset(train_ds), ConcatDataset(val_ds)

## Data loader preparation
local_batch_size = config["batch_size"] // n_gpus
train_loader = DataLoader(train_ds, shuffle=True, batch_size=local_batch_size, num_workers=0)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=local_batch_size, num_workers=0)

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

    if "node_kwargs" in config["input_constructor_kwargs"]:
        if "feature_extractor_path" in config["input_constructor_kwargs"]["node_kwargs"]:
            config["input_constructor_kwargs"]["node_kwargs"]["checkpoint_idx"] = i

    # Initialize classifier
    model = DenseLCClassifier(
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
        # strategy=DDPStrategy(find_unused_parameters=False),
        strategy=DDPStrategy(find_unused_parameters=True),
        # precision="16-mixed",
        logger=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename=f"{i}",
                save_top_k=1,
                monitor="val_loss", 
                mode="min"
            )
        ]
    )

    # Train
    trainer.fit(model, train_loader, val_loader)