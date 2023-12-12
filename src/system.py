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


def set_seed(seed: int) -> torch.Generator:
    # Try to make this implementation as deterministic as possible.
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class TrainingSystem(pl.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, any],
        wandb_logger,
    ):
        super().__init__()

        dataset_config = wandb_config["dataset_kwargs"]

        if wandb_config["architecture"] == "convlstm":
            self.model = src.networks.Seq2Seq(
                wandb_config=wandb_config,
            )
            self.model_config = wandb_config["convlstm_backbone_kwargs"]
        elif wandb_config["architecture"] == "ffn":
            self.model = src.networks.SimpleFeedForward(
                wandb_config=wandb_config,
            )
            self.model_config = wandb_config["ffn_backbone_kwargs"]
        elif wandb_config["architecture"] == "recurrent":
            self.model = src.networks.RNN(
                wandb_config=wandb_config,
            )
            self.model_config = wandb_config["recurrent_backbone_kwargs"]
        else:
            raise NotImplementedError

        self.reservoir_config = wandb_config["reservoir_kwargs"]
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
        precision = (torch.round(preds) * labels).sum() / torch.round(preds).sum()
        recall = (torch.round(preds) * labels).sum() / labels.sum()
        f1 = 2 * precision * recall / (precision + recall)

        result = {
            "loss": loss,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        return result

    def training_step(self, batch, batch_idx):
        obs, targs = batch
        preds = self.model(obs)
        result = self._calculate_loss(preds=preds, labels=targs)

        self.log(
            f"train/loss",
            result["loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"train/acc",
            result["acc"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"train/precision",
            result["precision"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"train/recall",
            result["recall"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"train/f1",
            result["f1"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # I want to log the beta parameter in each reservoir of the dam
        for l in self.model_config["reservoir_layers"]:
            betas = getattr(self.model.sequential, f"reservoir{l}").betas.detach().cpu()
            for i, beta in enumerate(betas):
                self.log(
                    f"train/layer{l}_dam{i}",
                    beta,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        return result

    def validation_step(self, batch, batch_idx):
        obs, targs = batch
        preds = self.model(obs)
        result = self._calculate_loss(preds=preds, labels=targs)
        self.log(
            f"val/loss",
            result["loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/acc",
            result["acc"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/precision",
            result["precision"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/recall",
            result["recall"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/f1",
            result["f1"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result

    def test_step(self, batch, num_generations=8):
        result = []
        for _ in range(num_generations):
            preds = self.model(batch)
            result.append(preds)
            batch = torch.cat([batch, preds.unsqueeze(0)], dim=-3)[:, :, 1:]
        return torch.cat(result, dim=-3)
