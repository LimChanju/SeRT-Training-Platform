from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


RISK_MODEL_VERSION = "risk_model_v0_hri_distance"
RISK_FEATURE_NAMES: tuple[str, ...] = (
    "robot_joint_pos",
    "gripper_width",
    "ee_pos",
    "cube_pos",
    "place_target_pos",
    "ee_to_cube",
    "cube_to_place_target",
    "task_phase",
    "controller_event",
    "controller_t",
    "human_head_pos",
    "human_left_hand_pos",
    "human_right_hand_pos",
    "ee_to_left_hand",
    "ee_to_right_hand",
    "min_hand_gripper_dist",
)


@dataclass(frozen=True)
class RiskThresholds:
    collision_dist_m: float = 0.03
    safe_dist_m: float = 0.12


class RiskMLP(nn.Module):
    """Small MLP that predicts an HRI risk / pseudo-ErrP score in [0, 1]."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(features))
