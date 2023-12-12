import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os


class ReservoirLayer(pl.LightningModule):
    def __init__(self, input_size, output_size, reservoir_config, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.leak_rate = reservoir_config["leak_rate"]
        transform_matrix_path = reservoir_config["path"]
        if os.path.exists(transform_matrix_path):
            self.transform_matrix = torch.load(
                transform_matrix_path
            )  # , map_location="cuda")
            # make self.stransorm_matrix a parameter
        else:
            with torch.no_grad():
                torch.manual_seed(int(transform_matrix_path[-4:-3]))
                sample = torch.randn((input_size, input_size))
                reservoir = (
                    sample
                    / torch.linalg.eigvals(sample).abs().max()
                    * reservoir_config["spectral_radius"]
                )
                self.transform_matrix = (
                    self.leak_rate * torch.eye(self.input_size)
                    + (1 - self.leak_rate) * reservoir
                )
                # save transform matrix
                torch.save(self.transform_matrix, transform_matrix_path)
                self.tranform_matrix.to("cuda")
        self.transform_matrix = nn.Parameter(self.transform_matrix, requires_grad=False)

        # self.readout = nn.Linear(input_size, output_size, bias=False)
        if reservoir_config["nonlinearity"] == "relu":
            self.activation_fn = F.relu
        elif reservoir_config["nonlinearity"] == "tanh":
            self.activation_fn = F.tanh
        else:
            self.activation_fn = lambda x: x

    def forward(self, x, u):
        u = u @ self.transform_matrix + x
        u = self.activation_fn(u)
        # u = self.readout(u)
        return u

    def __repr__(self):
        return f"ReservoirLayer(input_size={self.input_size}, output_size={self.output_size}, max_iter={self.max_iter}, leak_rate={self.leak_rate}, condition_number={self.condition_number})"


class ReservoirSkip(pl.LightningModule):
    def __init__(self, input_size, output_size, reservoir_config, **kwargs):
        super().__init__()

        self.input_size = input_size  # np.prod(input_shape)
        self.output_size = output_size  # np.prod(output_shape)
        self.reservoir_layer = ReservoirLayer(
            self.input_size, self.output_size, reservoir_config
        )
        self.skip_transform = None
        self.skip_dropout_prob = reservoir_config["skip_dropout_prob"]

        if reservoir_config["skip_transform_enabled"]:
            self.skip_transform = nn.Linear(input_shape, output_shape)
        else:
            self.skip_transform = lambda x: x

    def forward(self, x):
        batch_size = x.size(0)
        if len(x.shape) < 3:
            time_steps = 1
            x = x.unsqueeze(2)
        else:
            time_steps = x.shape[1]
        states = []
        u = torch.zeros_like(x[:, :, 0])
        for t in range(time_steps):
            x_t = x[:, :, t]
            x_flattened = x_t.reshape((batch_size, -1))
            u_flattened = u.reshape((batch_size, -1))
            u = self.reservoir_layer(x_flattened, u_flattened).reshape(x_t.shape)
            if self.skip_transform and torch.rand(1).item() > self.skip_dropout_prob:
                # is timewise dropout really the play?
                states.append(u + x_t)
            else:
                states.append(u)
        return torch.stack(states, dim=2)

    @property
    def readout(self):
        return self.reservoir_layer.readout


class Dam(pl.LightningModule):
    def __init__(self, input_shape, output_shape, reservoir_config, **kwargs):
        super().__init__()

        if output_shape == -1:
            self.output_shape = (input_shape[0], 1, input_shape[-2], input_shape[-1])
        else:
            self.output_shape = output_shape
        self.input_shape = input_shape
        self.input_size = np.prod(self.input_shape[-2:])
        self.output_size = np.prod(self.output_shape[-2:])
        self.n_reservoirs = reservoir_config["n_reservoirs"]
        self.reservoirs = []
        for i in range(self.n_reservoirs):
            transform_matrix_path = (
                f"/home/ubuntu/imgphys/precompute/transform_matrix{i+1}.pt"
            )
            reservoir_config["path"] = transform_matrix_path
            self.reservoirs.append(
                ReservoirSkip(self.input_size, self.output_size, reservoir_config)
            )
        self.beta_weights = nn.Parameter(
            torch.ones(self.n_reservoirs), requires_grad=True
        )
        self.collate_method = reservoir_config["collate_method"]

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, *self.input_shape))
        result = []
        for i, reservoir in enumerate(self.reservoirs):
            reservoir_output = self.beta_weights[i] * reservoir(x)
            reservoir_output.reshape((batch_size, *self.output_shape))
            result.append(reservoir_output)
        if self.collate_method == "sum":
            return torch.sum(torch.stack(result), dim=0).reshape(
                batch_size, *self.output_shape
            )
        elif self.collate_method == "mean":
            return torch.mean(torch.stack(result), dim=0).reshape(
                batch_size, *self.output_shape
            )
        elif self.collate_method == "concat":
            return torch.cat(result, dim=-1)
        else:
            raise NotImplementedError

    @property
    def betas(self):
        return self.beta_weights
