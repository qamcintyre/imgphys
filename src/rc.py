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
    def __init__(self, input_size, output_size, reservoir_config: Dict, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_iter = reservoir_config["max_iter"]
        self.leak_rate = reservoir_config["leak_rate"]
        self.reservoir = nn.Parameter(torch.randn(input_size, input_size))
        self.reservoir = self.reservoir / torch.linalg.eigvals(self.reservoir).max()*reservoir_config["spectral_radius"]
        self.condition_number = torch.linalg.cond(self.reservoir)
        self.readout = nn.Linear(input_size, output_size, bias=False)
        self.reservoir.requires_grad = False
        self.readout.requires_grad = True
    
    def forward(self, x):
        x = self.leak_rate*x + (1-self.leak_rate) * self.x @ self.reservoir
        for _ in range(self.max_iter):
            if torch.linalg.norm(x)/x.shape[0] < 1e-6:
                break
            x = self.leak_rate*x + (1-self.leak_rate) * self.x @ self.reservoir
        x = x @ self.readout
        return x
    

