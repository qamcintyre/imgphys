import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import json
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    # DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
import pprint
import wandb

from src.data import DataModule
from src.globals import default_config
from src.system import set_seed, ConvLSTMTrainingSystem


run = wandb.init(project="imgphys", config=default_config)
wandb_config = dict(wandb.config)

# Convert "None" (type: str) to None (type: NoneType)
for key in wandb_config.keys():
    if isinstance(wandb_config[key], str):
        if wandb_config[key] == "None":
            wandb_config[key] = None

# Create checkpoint directory for this run, and save the config to the directory.
run_checkpoint_dir = os.path.join("lightning_logs", wandb.run.id)
os.makedirs(run_checkpoint_dir)
with open(os.path.join(run_checkpoint_dir, "wandb_config.json"), "w") as fp:
    json.dump(obj=wandb_config, fp=fp)

# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.LearningRateMonitor.html
lr_monitor_callback = LearningRateMonitor(logging_interval="step", log_momentum=True)

torch_generator = set_seed(seed=wandb_config["seed"])


wandb_logger = WandbLogger(experiment=run)
checkpoint_callback = ModelCheckpoint(
    monitor="loss",
    save_top_k=1,
    mode="min",
    dirpath=run_checkpoint_dir
)

callbacks = [
    lr_monitor_callback,
    checkpoint_callback,  # Don't need to save these models.
]

datamodule = DataModule(
    wandb_config=wandb_config,
    run_checkpoint_dir=run_checkpoint_dir,
    torch_generator=torch_generator,
)


# Need to call this to determine the number of observation dimensions, and also initialize the memories.
datamodule.setup(stage="setup")

system = ConvLSTMTrainingSystem(wandb_config=wandb_config, wandb_logger=wandb_logger)
trainer = pl.Trainer(
    accumulate_grad_batches=wandb_config["accumulate_grad_batches"],
    callbacks=callbacks,
    check_val_every_n_epoch=wandb_config["check_val_every_n_epoch"],
    default_root_dir=run_checkpoint_dir,
    deterministic=True,
    # accelerator="gpu",
    # devices="4",
    # fast_dev_run=True,
    fast_dev_run=False,
    logger=wandb_logger,
    log_every_n_steps=1,
    # overfit_batches=1,  # useful for debugging
    gradient_clip_val=wandb_config["gradient_clip_val"],
    max_epochs=wandb_config["n_epochs"],
    num_sanity_val_steps=-1,  # -1 means runs all of validation before starting to train.
    # profiler="simple",  # Simplest profiler
    # profiler="advanced",  # More advanced profiler
    # profiler=PyTorchProfiler(filename=),  # PyTorch specific profiler
    precision=wandb_config["precision"],
)

# .fit() needs to be called below for multiprocessing.
# See: https://github.com/Lightning-AI/lightning/issues/13039
# See: https://github.com/Lightning-AI/lightning/discussions/9201
# See: https://github.com/Lightning-AI/lightning/discussions/151
if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    print("W&B Config:")
    pp.pprint(wandb_config)

    trainer.fit(
        model=system,
        datamodule=datamodule,
    )