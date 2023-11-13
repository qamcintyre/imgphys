"""
I am writing a pytorch lightning layer module for reservoir computing. It will take in an input_size n
and an output_size m, and will have a reservoir of size n. It will have a readout matrix of size m.
The reservoir will be a random nxn adjacency matrix and , with a nx1 input vector given in forward pass,
and an nxm readout layer. The reservoir will have a global stop gradient, but the readout matrix will
not. The forward pass will be max_iter steps of the reservoir, which is a layer param, followed by a readout matrix.
The forward pass will return the output of the readout matrix. The random reservoir must have eigenvalues less
than one, so that the reservoir does not explode.
"""

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

class ReservoirLayer(nn.Module):
    def __init__(self, input_size, output_size, reservoir_config, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_iter = reservoir_config["max_iter"]
        self.leak_rate = reservoir_config["leak_rate"]
        self.reservoir = nn.Parameter(torch.randn(input_size, input_size))
        self.reservoir = torch.nn.Parameter(self.reservoir / torch.linalg.eigvals(self.reservoir).abs().max() * reservoir_config["spectral_radius"])
        self.condition_number = torch.linalg.cond(self.reservoir)
        self.readout = nn.Linear(input_size, output_size, bias=False)
        self.reservoir.requires_grad = False
        self.readout.requires_grad = True
        if reservoir_config["nonlinearity"] == "relu":
            self.activation_fn = F.relu
        elif reservoir_config["nonlinearity"] == "tanh":
            self.activation_fn = F.tanh
        else:
            self.activation_fn = lambda x: x  
            
        self.transform_matrix = self.leak_rate*torch.eye(self.input_size)+(1-self.leak_rate)*self.reservoir

    def forward(self, x):
        x = x @ self.transform_matrix
        x = self.activation_fn(x)
        for _ in range(self.max_iter):
            if torch.linalg.norm(x) / x.shape[0] < 1e-6:
                break
            x = x @ self.transform_matrix
            x = self.activation_fn(x)
        x = x @ self.readout
        return x
    
class ReservoirSkip(nn.Module):
    def __init__(self, input_shape, output_shape, reservoir_config, **kwargs):
        super().__init__()
        
        if output_shape == -1:
            output_shape = input_shape

        self.input_size = np.prod(input_shape)
        self.output_size = np.prod(output_shape)
        self.reservoir_layer = ReservoirLayer(self.input_size, self.output_size, reservoir_config)
        self.skip_transform = None
        self.skip_dropout_prob = reservoir_config["skip_dropout_prob"]
        
        if reservoir_config["skip_transform_enabled"]:
            self.skip_transform = nn.Linear(input_shape, output_shape)
        else:
            self.skip_transform = lambda x: x
    def forward(self, x):
        batch_size = x.size(0)
        time_steps = x.shape[2]
        states=[]
        for _ in range(time_steps):
            x_flattened = x[:,:,_].view(batch_size, -1)
            reservoir_output = self.reservoir_layer(x_flattened)
            if self.skip_transform and torch.rand(1).item() > self.skip_dropout_prob:
                #transformed_input = self.skip_transform(x)

                states.append(reservoir_output.view_as(x) + x)
            else:
                states.append(reservoir_output.view_as(x))
        return torch.stack(states, dim=2)
