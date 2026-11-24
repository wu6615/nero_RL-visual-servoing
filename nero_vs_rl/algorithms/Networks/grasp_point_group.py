"""GAP-RL grasp point-group encoder as an SB3 features extractor.

Port of GAP-RL ``Networks/pointnet.py::GraspPointAppGroup`` +
``rl_utils.py::CustomGraspPointGroupExtractor``.  Per-point MLP (= Conv1d k=1)
with LayerNorm, max-pool over the K points of each grasp, concatenate with the
reference gripper point feature and grasp centre, group MLP, then a *masked*
max-pool over the N grasps (padded slots are excluded; GAP-RL relied on zeros).
"""
from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

STATE_KEYS = ("tcp_pose", "gripper_pos", "action", "grasp_vis")


def _mlp(in_dim: int, dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for d in dims:
        layers += [nn.Linear(in_dim, d), nn.LayerNorm(d, eps=1e-6), nn.ReLU()]
        in_dim = d
    return nn.Sequential(*layers)


class GraspPointGroupEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, point_dims=(16, 32), group_dims=(64, 256)):
        super().__init__()
        self.point_mlp = _mlp(in_ch, list(point_dims))
        self.group_mlp = _mlp(3 + 2 * point_dims[-1], list(group_dims))
        self.out_dim = group_dims[-1]

    def forward(self, x_init: torch.Tensor, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x_init (B,K,3)  x (B,N,K,3)  valid (B,N) bool
        B, N, K, _ = x.shape
        pts = torch.cat([x_init.unsqueeze(1), x], dim=1)          # (B,N+1,K,3)
        f = self.point_mlp(pts).max(dim=2).values                   # (B,N+1,C)
        center = x.mean(dim=2)                                      # (B,N,3)
        g = torch.cat([f[:, 1:], f[:, :1].expand(-1, N, -1), center], dim=-1)
        g = self.group_mlp(g)                                       # (B,N,D)
        g = g.masked_fill(~valid.unsqueeze(-1), -1e4).max(dim=1).values
        any_valid = valid.any(dim=1, keepdim=True)
        return torch.where(any_valid, g, torch.zeros_like(g))


class GraspPointGroupExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_dim: int = 128,
                 point_dims=(16, 32), group_dims=(64, 256)):
        state_in = sum(int(observation_space[k].shape[0]) for k in STATE_KEYS)
        super().__init__(observation_space, features_dim=state_dim + group_dims[-1])
        self.encoder = GraspPointGroupEncoder(3, point_dims, group_dims)
        self.state_map = nn.Linear(state_in, state_dim)
        nn.init.xavier_uniform_(self.state_map.weight)
        nn.init.zeros_(self.state_map.bias)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = torch.cat([obs[k] for k in STATE_KEYS], dim=1)
        valid = obs["grasp_scores"] > 0
        pn = self.encoder(obs["origin_gripper_pts"], obs["gripper_pts_diff"], valid)
        return torch.cat([self.state_map(state), pn], dim=1)
