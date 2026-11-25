"""Batched reward and success rules: port of DynamicGraspEnv.evaluate / compute_dense_reward (v8, clean grasp)."""
from __future__ import annotations

import math

import torch

from ...utils.config import EnvConfig


def closing_penalty(cfg: EnvConfig, i: dict) -> torch.Tensor:
    w = cfg.reward
    pen = (w.close_misalign * (i["misalign"] / math.radians(45)).clamp_max(1)
           + w.close_tilt * ((i["tilt"] - math.radians(15)) / math.radians(30)).clamp(0, 1)
           + w.close_moving * (i["rel_speed"] / w.close_speed_ref).clamp_max(1))
    on = ~i["grasped"] & i["closing"] & (i["dist_t"] < w.close_near_m)
    return torch.where(on, pen, torch.zeros_like(pen))


def dense_reward(cfg: EnvConfig, i: dict, steps: torch.Tensor) -> torch.Tensor:
    w = cfg.reward
    b = lambda x: x.to(i["dist_t"].dtype)
    r = w.info_exist * b(i["visible"] | (i["dist_t"] < 0.06))
    r = r + w.approach_trans * (1 - torch.tanh(5 * i["dist_t"].clamp(0.01, 0.5)))
    r = r + w.approach_rot * (1 - torch.tanh(5 * i["dist_r"].clamp(0.01, 0.5)))
    g = w.grasp + w.lift * (1 - torch.tanh(5 * (cfg.lift_height_m - i["lift"]).clamp_min(0)))
    r = r + b(i["grasped"]) * g + b(i["grasped"] & i["lifted"] & i["robot_static"]) * w.static
    r = r + closing_penalty(cfg, i) + w.collision * b(i["collision"]) + w.time
    r = r + b(i["missed"] | i["fell"]) * w.time * (cfg.max_steps - steps).clamp_min(0)
    ok = w.success + torch.where(i["robot_static"], 0.0, w.hold_still).to(r.dtype)
    return torch.where(i["success"], ok, r) * w.scale
