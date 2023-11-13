# Adapted from https://github.com/bsaha205/clam/blob/main/parser.py.
from functools import partial
import lightning.pytorch as pl
import numpy as np
import os
import pandas as pd
import scipy.io
import torch
import torch.utils.data
from torch.utils.data import Dataset, Subset
from torch import Tensor
# import torchvision
from typing import Any, Dict, List, Tuple
# import src.utils
# import torchvision.transforms as transforms
# import torchvision.transforms.v2 as transforms_v2
# from data_generation.one_body_generator import load_motion_sequences

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        wandb_config: Dict,
        run_checkpoint_dir: str,
        torch_generator: torch.Generator = None,
        num_workers: int = None,
        precision: int = 32,
    ):
        super().__init__()
        self.wandb_config = wandb_config
        self.run_checkpoint_dir = (
            run_checkpoint_dir  # Necessary if we write data to disk.
        )
        self.torch_generator = torch_generator
                
        # Recommendation: https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html
        if num_workers is None:
            num_workers = max(4, os.cpu_count() // 4)  # heuristic
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None

        # Allegedly pinning memory saves time. Not sure what it does though.
        # self.pin_memory = torch.cuda.is_available()
        self.pin_memory = False

        self.datum_shape = None
        self.precision = precision

    def setup(self, stage: str):
        train_dataset = NBodyDataset(
            wandb_config=self.wandb_config,
            train=True
        )
        val_dataset = NBodyDataset(
            wandb_config=self.wandb_config,
            train=False
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.datum_shape = train_dataset.datum_shape
        print(f"PhysicsDataModule.setup(stage={stage}) called.")


    def train_dataloader(self):
        print("PhysicsDataModule.train_dataloader() called.")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.wandb_config["batch_size_train"],
            shuffle=True,
            num_workers=self.wandb_config["num_workers"],
            drop_last=True,
            pin_memory=self.pin_memory,
            # persistent_workers=False,
            generator=self.torch_generator,
            # prefetch_factor=self.wandb_config["prefetch_factor"],
)

    def val_dataloader(self):
        print("PhysicsDataModule.val_dataloader() called.")
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.wandb_config["batch_size_val"],
            shuffle=False,
            num_workers=self.wandb_config["num_workers"],
            drop_last=True,
            pin_memory=self.pin_memory,
            # persistent_workers=False,
            generator=self.torch_generator,
            prefetch_factor=self.wandb_config["prefetch_factor"],
)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished.
        print(f"TrajectoryDataModule.teardown(stage={stage}) called.")

class NBodyDataset(torch.utils.data.Dataset):
    def __init__(self, wandb_config: Dict[str, Any], train: bool):
        self.wandb_config = wandb_config
        self.data = torch.tensor(np.load(wandb_config["data_path"]), dtype=torch.float32)
        
        
        if train:
            self.data = self.data[:int(wandb_config['train_val_split']*self.data.shape[0])]
        else:
            self.data = self.data[int(wandb_config['train_val_split']*self.data.shape[0]):]
        
        self.datum_shape = self.data.shape[-2:]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx].unsqueeze(0)
        return datum[:, :-1], datum[:, -1]


