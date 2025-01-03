import argparse ## REMOVE?
from pathlib import Path
import logging 
import time 
import yaml 
import os 

import torch.nn as nn
from lightning.pytorch import seed_everything
seed_everything(42, workers=True)
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

import ctypes
ligbcc_s = ctypes.CDLL("libgcc_s.so.1")

import sys 
sys.path.append("Path/to/EmulationUtilities")
from _data_utils import DataModule
from _nn_config import DataConfig, ModelConfig, TrainingConfig
import _models


"""
See doc. at bottom to see list of modifications must be made, and which functions to run in which order. 
"""

logging.basicConfig(level=logging.INFO, format='%(message)s')

# load configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", 
                    type=Path, 
                    default="config.yaml",)

p = parser.parse_args()
with open(p.config_path, "r") as f:
    config = yaml.safe_load(f)


data_config  = DataConfig(**config["data"])
model_config = ModelConfig(**config["model"])
train_config = TrainingConfig(**config["training"])


# load dataset module
data_module = DataModule(
    feature_columns = data_config.feature_columns,
    label_columns   = data_config.label_columns,
    batch_size      = data_config.batch_size,
    num_workers     = data_config.num_workers if data_config.num_workers is not None else os.cpu_count(),
    shuffle         = data_config.shuffle,
    feature_scaler  = data_config.feature_scaler,
    label_scaler    = data_config.label_scaler,
)
data_module.setup(
    train_data_path = Path(data_config.train_data_path),
    val_data_path   = Path(data_config.val_data_path),
    test_data_path  = Path(data_config.test_data_path),
    autoencoder     = model_config.autoencoder,
)

model = getattr(_models, model_config.type)(
    n_features=data_module.train_data.x.shape[-1],
    output_dim=data_module.train_data.y.shape[-1],
    **dict(model_config),
    )
# Change weight init
def init_weights_small_variance(m,):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=1.e-3)
        if model_config.zero_bias:
            m.bias.data.fill_(0) 

model.apply(init_weights_small_variance)

callbacks = []
callbacks.append(EarlyStopping(
    monitor  = "loss/val", 
    patience = train_config.stopping_patience, 
    mode     = "min", 
    verbose  = False,
    ))

if train_config.use_swa:
    callbacks.append(
        StochasticWeightAveraging(
            swa_lrs         = model_config.learning_rate,
            swa_epoch_start = train_config.swa_start,
            )
        )
    
    
# logger_name = f"time_test"
if train_config.log_save_dir is not None:
    logger_ = TensorBoardLogger(
        save_dir = train_config.default_root_dir, 
        name     = train_config.log_save_dir,
        # version  = 0 # Set manually if needed
        )
else:
    logger_ = TensorBoardLogger(
        save_dir = train_config.default_root_dir,
    )


trainer = pl.Trainer(
    callbacks           = callbacks,
    devices             = 1,
    # accelerator         = 'cpu',
    max_epochs          = train_config.max_epochs,
    default_root_dir    = train_config.default_root_dir,
    gradient_clip_val   = train_config.gradient_clip_val if train_config.gradient_clip_val != 0.0 else None,
    logger              = logger_,
    # log_every_n_steps   = 5,
    )

scalers_path = Path(trainer.logger.log_dir)
scalers_path.mkdir(parents=True, exist_ok=True)
data_module.dump_scalers(path=scalers_path / "scalers.pkl")

# Store config file used to run it
with open(scalers_path / "config.yaml", "w") as f:
    yaml.dump(config, f)


"""
Modifications to make:
    - At the top, use path correct Path to EmulationUtilities:
        sys.path.append("Path/to/EmulationUtilities")

Before running:
    - Update the config.yaml file according to the description in the README.md
"""


if __name__ == "__main__":
    ### Train the model 
    print("Starting training")
    t0 = time.time()
    # Perform training 
    trainer.fit(model=model, 
                datamodule=data_module,
                )

        
    dur = time.time() - t0
    dur_str = f"{dur//3600:.0f}hrs {(dur%3600)//60:.0f}min {dur%60:.0f}sec"
    print(f"Model: {trainer.logger.log_dir} took {dur_str} to train")

    with open(scalers_path / "training_duration.txt", "w") as f:
        f.write(f"{dur_str}\n")