from __future__ import annotations

import torch
from torch import nn


class MLPPolicy(nn.Module):
    """Simple tanh-squashed MLP policy for normalized 5D task actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MLPRegressor(nn.Module):
    """Simple MLP regressor for normalized continuous targets."""

    def __init__(
        self,
        obs_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
