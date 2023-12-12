import torch
import torch.nn as nn
from src.rc import ReservoirSkip, Dam
import lightning.pytorch as pl


# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(pl.LightningModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, activation, frame_size
    ):
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        nn.init.xavier_uniform_(self.W_ci)
        nn.init.xavier_uniform_(self.W_co)
        nn.init.xavier_uniform_(self.W_cf)

    def forward(self, X, H_prev, C_prev):
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(pl.LightningModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, activation, frame_size
    ):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(
            in_channels, out_channels, kernel_size, padding, activation, frame_size
        )

    def forward(self, X):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(
            batch_size, self.out_channels, seq_len, height, width, device=self.device
        )

        # Initialize Hidden State
        H = torch.zeros(
            batch_size, self.out_channels, height, width, device=self.device
        )

        # Initialize Cell Input
        C = torch.zeros(
            batch_size, self.out_channels, height, width, device=self.device
        )

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output


class Seq2Seq(pl.LightningModule):
    def __init__(self, wandb_config):
        super(Seq2Seq, self).__init__()

        model_config = wandb_config["convlstm_backbone_kwargs"]
        dataset_config = wandb_config["dataset_kwargs"]

        self.num_channels = dataset_config["channels"]
        self.num_kernels = model_config["n_kernels"]
        self.kernel_size = model_config["kernel_size"]
        self.padding = model_config["padding"]
        self.activation = model_config["activation"]
        self.frame_size = dataset_config["frame_size"]
        self.num_layers = model_config["num_layers"]
        self.num_frames_prediction = dataset_config["num_frames_prediction"]

        self.reservoir_config = wandb_config["reservoir_kwargs"]

        self.sequential = nn.Sequential()

        # Add rest of the layers
        for l in range(1, self.num_layers + 1):
            if l in model_config["reservoir_layers"]:
                self.sequential.add_module(
                    f"reservoir{l}",
                    Dam(
                        input_shape=(
                            self.num_kernels if l > 1 else self.num_channels,
                            self.num_frames_prediction,
                            *self.frame_size,
                        ),
                        output_shape=-1,
                        reservoir_config=self.reservoir_config,
                    ),
                )

            self.sequential.add_module(
                f"convlstm{l}",
                ConvLSTM(
                    in_channels=self.num_kernels if l > 1 else self.num_channels,
                    out_channels=self.num_kernels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    frame_size=self.frame_size,
                ),
            )

            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=self.num_kernels)
            )

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=self.num_kernels,
            out_channels=self.num_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])

        return nn.Sigmoid()(output)


class RNN(pl.LightningModule):
    def __init__(self, wandb_config):
        super(RNN, self).__init__()

        model_config = wandb_config["recurrent_backbone_kwargs"]
        dataset_config = wandb_config["dataset_kwargs"]

        self.num_channels = dataset_config["channels"]
        self.activation = model_config["activation"]
        self.frame_size = dataset_config["frame_size"]
        self.num_layers = model_config["num_layers"]
        self.hidden_size = torch.prod(torch.tensor(self.frame_size))
        self.num_frames_prediction = dataset_config["num_frames_prediction"]

        self.reservoir_config = wandb_config["reservoir_kwargs"]

        self.sequential = nn.Sequential()

        if 1 in model_config["reservoir_layers"]:
            self.sequential.add_module(
                f"dam1",
                Dam(
                    input_shape=(1, self.num_frames_prediction, *self.frame_size),
                    output_shape=-1,
                    reservoir_config=self.reservoir_config,
                ),
            )
        elif len(model_config["reservoir_layers"]) > 0:
            raise NotImplementedError

        self.rnn = nn.RNN(
            input_size=torch.prod(torch.tensor(self.frame_size)),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity=self.activation,
            batch_first=True,
        )

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)
        output = output.reshape(output.shape[0], output.shape[2], -1)
        output, _ = self.rnn(output)
        output = nn.Sigmoid()(output)
        output = output.reshape(output.shape[0], output.shape[1], *self.frame_size)

        return output


class SimpleFeedForward(pl.LightningModule):
    def __init__(self, wandb_config):
        super(SimpleFeedForward, self).__init__()

        model_config = wandb_config["ffn_backbone_kwargs"]
        dataset_config = wandb_config["dataset_kwargs"]

        self.num_channels = dataset_config["channels"]

        self.frame_height, self.frame_width = dataset_config["frame_size"]

        self.input_size = (
            dataset_config["num_frames_prediction"]
            * dataset_config["channels"]
            * self.frame_height
            * self.frame_width
        )

        self.output_size = (
            dataset_config["channels"] * self.frame_height * self.frame_width
        )
        self.num_layers = model_config["num_layers"]
        self.activation = model_config["activation"]
        self.seq = nn.Sequential()

        self.reservoir_config = wandb_config["reservoir_kwargs"]
        for l in range(1, self.num_layers + 1):
            if l in model_config["reservoir_layers"]:
                if l == 1:
                    self.seq.add_module(
                        f"reservoir{l}",
                        Dam(
                            input_shape=(
                                dataset_config["num_frames_prediction"],
                                int(
                                    self.input_size
                                    / dataset_config["num_frames_prediction"]
                                ),
                            ),
                            output_shape=-1,
                            reservoir_config=self.reservoir_config,
                        ),
                    )
                else:
                    self.seq.add_module(
                        f"reservoir{l}",
                        Dam(
                            input_shape=(512,),
                            output_shape=-1,
                            reservoir_config=self.reservoir_config,
                        ),
                    )

            if l == 1:
                self.seq.add_module("flatten", nn.Flatten())
                self.seq.add_module(f"fc{l}", nn.Linear(self.input_size, 512))
            elif l == self.num_layers:
                self.seq.add_module(f"fc{l}", nn.Linear(512, self.output_size))
            else:
                self.seq.add_module(f"fc{l}", nn.Linear(512, 512))
            if l != self.num_layers:
                if self.activation == "tanh":
                    self.seq.add_module(f"tanh{l}", nn.Tanh())
                elif self.activation == "relu":
                    self.seq.add_module(f"relu{l}", nn.ReLU())
                else:
                    raise NotImplementedError

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.seq(x)
        x = x.view(-1, self.num_channels, self.frame_height, self.frame_width)
        x = nn.Sigmoid()(x)
        return x
