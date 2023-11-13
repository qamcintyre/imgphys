# Partially adapted from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
from copy import deepcopy
import math
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    # DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)

from lightning.pytorch.utilities import grad_norm
import os
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
import tqdm
from typing import Any, Callable, Dict, List, Tuple, Union

import src.data
import src.networks
# import torchvision.transforms as transforms
# import torchvision.transforms.v2 as transforms_v2


class LSTMReservoirTrainingSystem(pl.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, any],
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lr = finetune_learning_rate
        self.learning_rate_scheduler = finetune_learning_rate_scheduler
        self.weight_decay = finetune_weight_decay
        self.max_finetune_epochs = max_finetune_epochs
        # Mapping from representation h to classes
        # TODO: Replace with LazyLinear; maybe not necessary?
        self.readout = nn.Linear(
            in_features=feature_dim, out_features=num_classes, bias=True
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if (
            self.learning_rate_scheduler is None
            or self.learning_rate_scheduler == "None"
        ):
            return [optimizer]
        elif self.learning_rate_scheduler == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.max_finetune_epochs,
            )
        else:
            raise NotImplementedError
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        feats, labels = batch
        preds = self.readout(feats)
        loss = torch.nn.functional.cross_entropy(input=preds, target=labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(
            f"finetune/{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"finetune/{mode}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

def set_seed(seed: int) -> torch.Generator:
    # Try to make this implementation as deterministic as possible.
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

class ConvLSTMTrainingSystem(pl.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, any],
        wandb_logger,
    ):
        super().__init__()
        
        model_config = wandb_config["backbone_kwargs"]
        dataset_config = wandb_config["dataset_kwargs"]
        
        self.model = src.networks.Seq2Seq(
            wandb_config=wandb_config,
        )

        self.lr = wandb_config["lr"]
        self.learning_rate_scheduler = wandb_config["learning_rate_scheduler"]
        self.weight_decay = wandb_config["weight_decay"]
        
        self.wandb_logger = wandb_logger
        
        self.configure_optimizers()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if (
            self.learning_rate_scheduler is None
            or self.learning_rate_scheduler == "None"
        ):
            return [optimizer]
        else:
            raise NotImplementedError
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, preds, labels):
        loss = nn.BCELoss()(preds, labels)
        acc = (torch.round(preds) == labels).float().mean()
        self.log(
            f"loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"acc",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss, acc

    def training_step(self, batch, batch_idx):
        obs, targs = batch
        preds = self.model(obs)
        return self._calculate_loss(preds = preds, labels = targs)

    def validation_step(self, batch, batch_idx):
        obs, targs = batch
        preds = self.model(obs)
        return self._calculate_loss(preds = preds, labels = targs)

    def test_step(self, batch, num_generations=10):
        for _ in range(num_generations):
            preds = self.model(batch)
            batch = torch.cat([batch, preds.unsqueeze(0)], dim=-3)
        return batch