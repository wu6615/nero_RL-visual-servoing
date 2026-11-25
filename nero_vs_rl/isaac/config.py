"""Isaac-only settings (isaac.yaml); the task itself uses the shared EnvConfig."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import ISAAC_CONFIG


@dataclass
class DriveConfig:
    stiffness: list[float]
    damping: list[float]
    effort: float
    armature: float
    joint_friction: float
    gripper_stiffness: float
    gripper_damping: float
    gripper_effort: float


@dataclass
class IsaacTrainConfig:
    total_timesteps: int
    learning_rate: float
    buffer_size: int
    batch_size: int
    gamma: float
    tau: float
    learning_starts: int
    gradient_steps: int
    checkpoint_every: int
    success_window: int
    net_arch: list[int] = field(default_factory=lambda: [256, 256])


@dataclass
class IsaacConfig:
    num_envs: int
    env_spacing: float
    physics_dt: float
    object_library: int
    object_seed: int
    belt_friction: float
    finger_friction: float
    drive: DriveConfig
    train: IsaacTrainConfig


def load_isaac_config(path: Path = ISAAC_CONFIG) -> IsaacConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return IsaacConfig(**{**d, "drive": DriveConfig(**d["drive"]), "train": IsaacTrainConfig(**d["train"])})
