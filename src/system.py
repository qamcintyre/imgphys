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
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2


class ReservoirTrainingSYstem(pl.LightningModule):
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


class MultiViewSSLTrainingSystem(pl.LightningModule):
    def __init__(self, wandb_config: Dict, wandb_logger):
        # super().__init__f()
        super().__init__()

        # Should save hyperparameters to checkpoint.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html
        # self.save_hyperparameters()

        self.wandb_config = wandb_config
        self.wandb_logger = wandb_logger

        self.ssl_framework = wandb_config["ssl_framework"]
        if self.ssl_framework == "alignment_and_uniformity":
            # https://arxiv.org/abs/2005.10242
            ssl_system_constructor = AlignmentAndUniformitySSLSystem
        elif self.ssl_framework == "mmcr":
            # https://arxiv.org/abs/2303.03307
            ssl_system_constructor = MMCRSSLSystem
        elif self.ssl_framework == "mmcr_avg":
            ssl_system_constructor = MMCRAverageSSLSystem
        elif self.ssl_framework == "simclr":
            # https://arxiv.org/abs/2002.05709
            ssl_system_constructor = SimCLRSSLSystem
        elif self.ssl_framework == "swav":
            # ssl_system_constructor = SwaVSSLSystem
            raise NotImplementedError
        elif self.ssl_framework == "tico":
            # https://arxiv.org/abs/2206.10698
            ssl_system_constructor = TICOSystem
        elif self.ssl_framework == "vicreg":
            # https://arxiv.org/abs/2105.04906
            ssl_system_constructor = VICRegSSLSystem
        elif self.ssl_framework == "vicreg_mmcr":
            ssl_system_constructor = VICRegMMCRSSLSystem
        elif self.ssl_framework == "w_mse":
            # https://arxiv.org/abs/2007.06346
            # ssl_system_constructor = WMSESSLSystem
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown ssl_framework: {self.ssl_framework}")

        # For pairwise SSL frameworks, confirm that we are using 2 views.
        if self.ssl_framework in [
            "alignment_and_uniformity",
            "simclr",
            "tico",
            "vicreg",
        ]:
            assert wandb_config["n_views"] == 2

        self.ssl_system = ssl_system_constructor(
            wandb_config=wandb_config,
        )

    def configure_optimizers(self) -> Dict:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        # TODO: Maybe add SWA
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging
        if self.wandb_config["pretrain_optimizer"] == "adadelta":
            optimizer = torch.optim.Adadelta(
                self.parameters(),
                lr=self.wandb_config["pretrain_learning_rate"],
                weight_decay=self.wandb_config["pretrain_weight_decay"],
            )
        elif self.wandb_config["pretrain_optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.wandb_config["pretrain_learning_rate"],
                weight_decay=self.wandb_config["pretrain_weight_decay"],
                eps=1e-4,  # https://stackoverflow.com/a/42420014/4570472
            )
        elif self.wandb_config["pretrain_optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.wandb_config["pretrain_learning_rate"],
                weight_decay=self.wandb_config["pretrain_weight_decay"],
                eps=1e-4,  # https://stackoverflow.com/a/42420014/4570472
            )
        elif self.wandb_config["pretrain_optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.wandb_config["pretrain_learning_rate"],
                weight_decay=self.wandb_config["pretrain_weight_decay"],
                momentum=0.9,
                eps=1e-4,
            )
        elif self.wandb_config["pretrain_optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.wandb_config["pretrain_learning_rate"],
                weight_decay=self.wandb_config["pretrain_weight_decay"],
                momentum=0.9,
            )
        else:
            # TODO: add adafactor https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            raise NotImplementedError(f"{self.wandb_config['optimizer']}")

        optimizer_and_maybe_others_dict = {
            "optimizer": optimizer,
        }

        if self.wandb_config["pretrain_learning_rate_scheduler"] is None:
            pass
        elif (
            self.wandb_config["pretrain_learning_rate_scheduler"]
            == "cosine_annealing_warm_restarts"
        ):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=2,
            )
            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler

        elif (
            self.wandb_config["pretrain_learning_rate_scheduler"]
            == "linear_warmup_cosine_annealing"
        ):
            from flash.core.optimizers import LinearWarmupCosineAnnealingLR

            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=1,
                max_epochs=self.wandb_config["n_epochs"],
            )

            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler

        elif (
            self.wandb_config["pretrain_learning_rate_scheduler"]
            == "reduce_lr_on_plateau"
        ):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                factor=0.95,
                optimizer=optimizer,
                patience=3,
            )
            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
            optimizer_and_maybe_others_dict["monitor"] = "train/loss=total_loss"
        else:
            raise NotImplementedError(f"{self.wandb_config['learning_rate_scheduler']}")

        return optimizer_and_maybe_others_dict

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        observations = batch[0]
        forward_results = self.ssl_system.forward(x=observations)
        loss_results = self.ssl_system.compute_loss(
            forward_results=forward_results,
        )
        for loss_str, loss_val in loss_results.items():
            self.log(
                f"train/{loss_str}",
                loss_val,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        return loss_results["total_loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        observations = batch[0]
        forward_results = self.ssl_system.forward(
            x=observations,
        )
        loss_results = self.ssl_system.compute_loss(
            forward_results=forward_results,
        )

        for loss_str, loss_val in loss_results.items():
            self.log(
                f"val/{loss_str}",
                loss_val,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )


class SSLSystem(pl.LightningModule):
    def __init__(self, wandb_config: Dict[str, any], **kwargs):
        super().__init__()
        self.wandb_config = wandb_config
        self.backbone = self.create_backbone(wandb_config=wandb_config)
        self.projection = self.create_projection(wandb_config=wandb_config)
        self.save_hyperparameters()

    @staticmethod
    def create_backbone(wandb_config: Dict[str, any]):
        # Create network backbone.
        backbone_kwargs = wandb_config["backbone_kwargs"]
        if backbone_kwargs["architecture"] == "identity":
            backbone = src.networks.IdentityNetwork(wandb_config=wandb_config)
        elif backbone_kwargs["architecture"] == "mlp":
            backbone = src.networks.MLPNetwork(
                **backbone_kwargs,
            )
        else:
            raise ValueError("Unknown network: {}".format(wandb_config["model"]))
        return backbone

    @staticmethod
    def create_projection(wandb_config: Dict[str, any]):
        # Create network backbone.
        projection_kwargs = wandb_config["projection_kwargs"]
        if projection_kwargs["architecture"] == "identity":
            projection = src.networks.IdentityNetwork(wandb_config=wandb_config)
        elif projection_kwargs["architecture"] == "mlp":
            # if self.wandb_config["backbone_kwargs"]["layer_widths"] is None:
            #     datum_shape = self.wandb_config["datum_shape"]
            # else:
            #     datum_shape = self.wandb_config["backbone_kwargs"]["layer_widths"][-1]
            projection = src.networks.MLPNetwork(
                **projection_kwargs,
            )
        else:
            raise ValueError("Unknown network: {}".format(wandb_config["model"]))
        return projection

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        :param x: Shape: (batch size, n_views, n observation dims...)
        """

        backbone_results = self.backbone.forward(x=x)
        projection_results = self.projection.forward(
            x=backbone_results["outputs"],
        )

        forward_results = {
            "embeddings": backbone_results["outputs"],
            "projections": projection_results["outputs"],
        }
        return forward_results


class AlignmentAndUniformitySSLSystem(SSLSystem):
    # https://arxiv.org/abs/2005.10242
    def __init__(
        self,
        wandb_config: Dict[str, any],
        uniformity_prefactor: float = 0.5,
        temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__(wandb_config=wandb_config, **kwargs)
        self.uniformity_prefactor = uniformity_prefactor
        self.temperature = temperature

        from lightly.loss.hypersphere_loss import HypersphereLoss

        self.loss_fn = HypersphereLoss(
            t=self.temperature,
            lam=self.uniformity_prefactor,
        )

    def compute_loss(
        self, forward_results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        projection_view_one = forward_results["projections"][:, 0, :]
        projection_view_two = forward_results["projections"][:, 1, :]

        # The loss function internally normalizes the projections so we don't have to.
        total_loss = self.loss_fn(
            projection_view_one,
            projection_view_two,
        )

        loss_results = {
            "total_loss": total_loss,
        }
        return loss_results

    def compute_alignment_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).norm(dim=1).pow(self.power).mean()

    def compute_uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-self.temperature).exp().mean().log()


class MMCRSSLSystem(SSLSystem):
    # https://arxiv.org/abs/2303.03307
    def compute_loss(
        self, forward_results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch size, number of views, projection dimension)
        normalized_projections = torch.nn.functional.normalize(
            input=forward_results["projections"],
            dim=2,
        )

        # Average over views/transformations/augmentations.
        # Shape: (batch size, embedding dim)
        centroid_matrix = torch.mean(normalized_projections, dim=1)

        # This is just for logging/debugging/understanding purposes.
        average_centroid_norm = centroid_matrix.norm(p=2, dim=1).mean()

        # Compute nuclear norm of centroid matrix.
        nuclear_norm = centroid_matrix.norm(p="nuc")
        neg_nuclear_norm = -nuclear_norm

        loss_results = {
            "total_loss": neg_nuclear_norm,
            "mmcr_loss": neg_nuclear_norm,
            "nuclear_norm": nuclear_norm,
            "average_centroid_norm": average_centroid_norm,
        }
        return loss_results


class MMCRAverageSSLSystem(SSLSystem):
    def compute_loss(
        self, forward_results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        normalized_projections = torch.nn.functional.normalize(
            input=forward_results["projections"],
            dim=2,
        )

        # Average over views.
        # Shape: (batch size, embedding dim)
        centroid_matrix = torch.mean(normalized_projections, dim=1)

        # Maximize the squared norms of each row.
        # Shape: (batch size,)
        squared_norms_of_means = centroid_matrix.norm(p=2, dim=1).pow(2)
        # Shape: (,)
        reconstruction_loss = -squared_norms_of_means.mean()

        # Minimize the squared norm of the mean of the means.
        # Shape: (,)
        uniformity_loss = centroid_matrix.mean(dim=0).norm(p=2).pow(2)

        total_loss = uniformity_loss + reconstruction_loss

        loss_results = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "uniformity_loss": uniformity_loss,
        }
        return loss_results


class SimCLRSSLSystem(SSLSystem):
    # https://arxiv.org/abs/2002.05709
    def __init__(
        self,
        wandb_config: Dict[str, any],
        temperature: float = 1.0,
        weights: float = 1.0,
        **kwargs,
    ):
        super().__init__(wandb_config=wandb_config, **kwargs)
        self.temperature = temperature
        self.weights = weights

        from lightly.loss.ntx_ent_loss import NTXentLoss

        self.loss_fn = NTXentLoss(
            temperature=self.temperature,
            memory_bank_size=0,  # Use 0 for SimCLR. Use 4096 for MoCo.
            gather_distributed=True,  # TODO: Investigate whether this is correct.
        )

    def compute_loss(
        self, forward_results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        projections_view_one = forward_results["projections"][:, 0, :]
        projections_view_two = forward_results["projections"][:, 1, :]

        # Lightly's NTXentLoss internally normalizes the projections, so we don't need to.
        loss = self.loss_fn(
            projections_view_one,
            projections_view_two,
        )

        losses_results = {
            "total_loss": loss,
        }

        return losses_results


class TICOSystem(SSLSystem):
    def __init__(
        self,
        wandb_config: Dict[str, any],
        ema_covariance: float = 0.9,
        cov_prefactor: float = 20.0,
    ):
        super().__init__(wandb_config=wandb_config)
        self.covariance_ema_update = ema_covariance
        self.cov_prefactor = cov_prefactor

        from lightly.loss.tico_loss import TiCoLoss

        self.loss_fn = TiCoLoss(
            beta=self.covariance_ema_update,
            rho=self.cov_prefactor,
            gather_distributed=True,
        )

    def compute_loss(
        self,
        forward_results: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        projections = forward_results["projections"]
        projection_view_one = projections[:, 0, :]
        projection_view_two = projections[:, 1, :]

        total_loss = self.loss_fn(projection_view_one, projection_view_two)

        losses_results = {
            "total_loss": total_loss,
        }

        return losses_results


class VICRegSSLSystem(SSLSystem):
    def __init__(
        self,
        wandb_config: Dict[str, any],
        inv_prefactor: float = 25.0,
        var_prefactor: float = 25.0,
        cov_prefactor: float = 1.0,
        epsilon: float = 1e-4,
        **kwargs,
    ):
        super().__init__(wandb_config=wandb_config, **kwargs)
        self.inv_prefactor = inv_prefactor
        self.var_prefactor = var_prefactor
        self.cov_prefactor = cov_prefactor
        self.epsilon = epsilon

        from lightly.loss.vicreg_loss import VICRegLoss

        self.loss_fn = VICRegLoss(
            lambda_param=self.inv_prefactor,
            mu_param=self.var_prefactor,
            nu_param=self.cov_prefactor,
            gather_distributed=True,
            eps=self.epsilon,
        )

    def compute_loss(
        self,
        forward_results: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        projections = forward_results["projections"]
        projection_view_one = projections[:, 0, :]
        projection_view_two = projections[:, 1, :]

        total_loss = self.loss_fn(projection_view_one, projection_view_two)

        losses_results = {
            "total_loss": total_loss,
        }

        return losses_results


class VICRegMMCRSSLSystem(SSLSystem):
    def compute_loss(
        self, forward_results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Average over views.
        centroid_matrix = torch.mean(forward_results["projections"], dim=1)

        # Returned in order.
        S = torch.linalg.svdvals(centroid_matrix)

        if centroid_matrix.shape[0] < centroid_matrix.shape[1]:
            upper_bound = float(centroid_matrix.shape[0])
        else:
            upper_bound = math.sqrt(centroid_matrix.shape[0] * centroid_matrix.shape[1])

        truncated_S = torch.nn.functional.relu(upper_bound - S)

        loss_results = {
            "total_loss": truncated_S.sum(),
        }
        return loss_results


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


# TODO: Might need to investigate this
# https://lightning.ai/forums/t/ddp-replacing-torch-dist-calls-with-pl-directives-for-inter-node-communication/2953
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

    def handle_sigusr1(self, signum, frame):
        # Note: Rylan added a self argument to this function. Unsure if correct.
        os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
        exit()

    def handle_sigterm(self, signum, frame):
        # Note: Rylan added a self argument to this function. Unsure if correct.
        pass


def exclude_bias_and_norm(p):
    return p.ndim == 1


def extract_off_diagonal_elements(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@torch.no_grad()
def embed_data_using_backbone(
    wandb_config: Dict[str, Any], backbone: pl.LightningModule
) -> Dict[str, torch.utils.data.TensorDataset]:
    print("Embedding data using backbone...")
    train_dataset, val_dataset, _ = src.data.create_datasets(
        dataset=wandb_config["finetune_dataset"],
        dataset_dir=wandb_config["dataset_dir"],
        dataset_kwargs=wandb_config["finetune_dataset_kwargs"],
        n_views=1,
        sample_percent=wandb_config["finetune_dataset_sample_percent"],
        seed=wandb_config["seed"],
        **wandb_config["finetune_dataset_kwargs"],
    )

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    embedded_datasets_by_split = {}
    for split, dataset in [("train", train_dataset), ("val", val_dataset)]:
        # Encode all images
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=2,
            shuffle=False,
            drop_last=False,
        )
        # TODO: how should augmentations be handled here?
        embeddings, labels = [], []
        for batch_imgs, batch_labels in tqdm.tqdm(data_loader):
            batch_imgs = batch_imgs.to(device)
            batch_embeddings = backbone(batch_imgs)["outputs"]
            # The second dimension is the number of views, which is set to 1. Remove it.
            embeddings.append(batch_embeddings[:, 0].detach().cpu())
            labels.append(batch_labels)
            # break  # useful for fast debugging.

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)

        embedded_datasets_by_split[split] = torch.utils.data.TensorDataset(
            embeddings, labels
        )

    print("Finished embedding data using backbone.")
    return embedded_datasets_by_split


def fit_and_evaluate_ssl_system(
    wandb_config: Dict[str, Any],
    wandb_logger,
    ssl_system: pl.LightningModule,
    run_checkpoint_dir: str,
    batch_size: int = 1024,
    finetune_epochs: int = 250,
    **kwargs,
) -> None:
    # backbone = deepcopy(ssl_system.backbone)
    embedded_data_by_split = embed_data_using_backbone(
        wandb_config=wandb_config, backbone=ssl_system.backbone
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        embedded_data_by_split["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,  # Without these: RuntimeError: DataLoader worker exited unexpectedly
    )

    # TODO: Should this be val or test?
    val_loader = torch.utils.data.DataLoader(
        embedded_data_by_split["val"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=0,  # Without these: RuntimeError: DataLoader worker exited unexpectedly
    )

    finetune_system = src.systems.MultiViewSSLAffineClassificationEvalSystem(
        feature_dim=embedded_data_by_split["train"].tensors[0].shape[1],
        num_classes=embedded_data_by_split["train"].tensors[1].max().item() + 1,
        max_finetune_epochs=finetune_epochs,
        finetune_learning_rate=wandb_config["finetune_learning_rate"],
        finetune_learning_rate_scheduler=wandb_config[
            "finetune_learning_rate_scheduler"
        ],
        finetune_weight_decay=wandb_config["finetune_weight_decay"],
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(run_checkpoint_dir, "affine_classification_eval"),
        accelerator="gpu",
        # devices=1,
        max_epochs=finetune_epochs,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="finetune/val_acc"
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=wandb_logger,
        # enable_progress_bar=False,
        check_val_every_n_epoch=10,
        # log_every_n_steps=1,
        profiler="simple",
    )
    trainer.logger._default_hp_metric = None

    trainer.validate(model=finetune_system, dataloaders=val_loader)
    trainer.fit(
        model=finetune_system,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


def set_seed(seed: int) -> torch.Generator:
    # Try to make this implementation as deterministic as possible.
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator