import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from pyknos.nflows.nn import nets

from typing import Any, Callable, Dict, Optional, Union
from copy import deepcopy
import numpy as np

from torch import Tensor, nn, optim
from torch.utils.data import TensorDataset, DataLoader


from sbi.utils.torchutils import atleast_2d


from time import time


class RegressionInference:
    """
    A bit unnecessary but copying the sbi/GBI API to be consistent.
    """

    def __init__(
        self,
        theta: Tensor,
        x: Tensor,
        predict_theta: bool,
        num_layers: int,
        num_hidden: int,
        net_type: str = "resnet",
        positive_constraint_fn: str = None,
        net_kwargs: Optional[Dict] = {},
    ):
        if predict_theta:
            self.predict_theta = True
            self.X, self.Y = x, theta
        else:
            self.predict_theta = False
            self.X, self.Y = theta, x

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.net_type = net_type
        self.net_kwargs = net_kwargs

        """Initialize neural network for regression."""
        self.init_network()

    def init_network(self, seed=0):
        torch.manual_seed(seed)
        self.network = RegressionNet(
            self.X.shape[1],
            self.Y.shape[1],
            self.num_layers,
            self.num_hidden,
            self.net_type,
            **self.net_kwargs,
        )

    def train(
        self,
        network: Optional[nn.Module] = None,
        training_batch_size: int = 500,
        max_n_epochs: int = 1000,
        stop_after_counter_reaches: int = 50,
        validation_fraction: float = 0.1,
        print_every_n: int = 20,
        plot_losses: bool = True,
    ) -> nn.Module:
        # Can use custom distance net, otherwise take existing in class.
        if network != None:
            if self.network != None:
                print("Warning: Overwriting existing network.")
            self.network = network

        # Define loss and optimizer.
        nn_loss = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters())

        # Train/val split
        dataset = TensorDataset(self.X, self.Y)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [1 - validation_fraction, validation_fraction]
        )
        dataloader = DataLoader(train_set, batch_size=training_batch_size, shuffle=True)
        X_val, Y_val = val_set[:]

        # Training loop.
        train_losses, val_losses = [], []
        epoch = 0
        self._val_loss = torch.inf
        while epoch <= max_n_epochs and not self._check_convergence(
            epoch, stop_after_counter_reaches
        ):
            time_start = time()
            for i_b, batch in enumerate(dataloader):
                optimizer.zero_grad()
                # Forward pass for distances.
                Y_pred = self.network(batch[0])

                # Training loss.
                l = nn_loss(batch[1], Y_pred)
                l.backward()
                optimizer.step()
                train_losses.append(l.detach().item())

            # Compute validation loss each epoch.
            with torch.no_grad():
                self._val_loss = (
                    nn_loss(Y_val, self.network(X_val).squeeze()).detach().item()
                )
                val_losses.append([(i_b + 1) * epoch, self._val_loss])

            # Print validation loss
            if epoch % print_every_n == 0:
                print(
                    f"{epoch}: train loss: {train_losses[-1]:.6f}, val loss: {self._val_loss:.6f}, best val loss: {self._best_val_loss:.6f}, {(time()-time_start):.4f} seconds per epoch."
                )

            epoch += 1

        print(f"Network converged after {epoch-1} of {max_n_epochs} epochs.")

        # Plot loss curves for convenience.
        self.train_losses = torch.Tensor(train_losses)
        self.val_losses = torch.Tensor(val_losses)
        if plot_losses:
            self._plot_losses(self.train_losses, self.val_losses)

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self.network.zero_grad(set_to_none=True)
        return deepcopy(self.network)

    def _check_convergence(self, counter: int, stop_after_counter_reaches: int) -> bool:
        """Return whether the training converged yet and save best model state so far.
        Checks for improvement in validation performance over previous batches or epochs.
        """
        converged = False

        assert self.network is not None
        network = self.network

        # (Re)-start the epoch count with the first epoch or any improvement.
        if counter == 0 or self._val_loss < self._best_val_loss:
            self._best_val_loss = self._val_loss
            self._counts_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(network.state_dict())
        else:
            self._counts_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._counts_since_last_improvement > stop_after_counter_reaches - 1:
            network.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(8, 3))
        plt.plot(train_losses, "k", alpha=0.8)
        plt.plot(val_losses[:, 0], val_losses[:, 1], "r.-", alpha=0.8)
        plt.savefig("losses.png")


class RegressionNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        hidden_features,
        net_type="resnet",
        activation=F.relu,
        dropout_prob=0.0,
        use_batch_norm=False,
        activate_output=False,
        in_transform=None,
    ):
        super().__init__()
        if net_type == "MLP":
            net = nets.MLP(
                in_shape=[input_dim],
                out_shape=[output_dim],
                hidden_sizes=[hidden_features] * num_layers,
                activation=activation,
                activate_output=activate_output,
            )

        elif net_type == "resnet":
            net = nets.ResidualNet(
                in_features=input_dim,
                out_features=output_dim,
                hidden_features=hidden_features,
                num_blocks=num_layers,
                activation=activation,
                dropout_probability=dropout_prob,
                use_batch_norm=use_batch_norm,
            )
        elif net_type == "linear":
            net = LinearRegression(input_dim, output_dim)
        else:
            raise NotImplementedError
        self.network = net

        if in_transform is not None:
            self.in_transform = in_transform
        else:
            self.in_transform = nn.Identity()

    def forward(self, x):
        x = self.in_transform(x)
        x = self.network(x)
        return x


class RegressionEnsemble:
    def __init__(self, neural_nets):
        if type(neural_nets) is not list:
            neural_nets = [neural_nets]
        self.n_ensemble = len(neural_nets)
        self.neural_nets = neural_nets

    def sample(self, x, sample_shape=None):
        # sample_shape is a dummy argument that makes it consistent with the other inference algorithms
        with torch.no_grad():
            return torch.stack([net(x) for net in self.neural_nets])


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
