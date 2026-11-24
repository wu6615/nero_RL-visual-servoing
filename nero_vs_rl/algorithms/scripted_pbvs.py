"""Scripted position-based visual servoing (PBVS) grasp baseline using only observed grasps.

Phases: approach (pre-grasp above the closest observed grasp, with velocity
feed-forward) -> descend -> close -> lift.  Uses the env's observed grasps
(``info['observed_grasps_ee']``), never ground truth, so it doubles as a
solvability check of the env and as a heuristic baseline like GAP-RL's
``test_heuristic_plan``.
"""
from __future__ import annotations

import numpy as np

from ..utils.config import EnvConfig
from ..utils.geometry import RZ_PI, logmap, rot_angle


class ScriptedGraspPolicy:
    def __init__(self, cfg: EnvConfig, pregrasp_m: float = 0.08, lead_steps: float = 1.5,
                 close_steps: int = 4, gain: float = 0.8):
        self.cfg = cfg
        self.pregrasp = pregrasp_m
        self.lead = lead_steps
        self.close_steps = close_steps
        self.gain = gain
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.counter = 0
        self.blind_steps = 0

    def __call__(self, obs: dict, info: dict) -> np.ndarray:
        return self.act(obs, info)

    def act(self, obs: dict, info: dict) -> np.ndarray:
        c = self.cfg
        dt = 1.0 / c.control_freq
        r6 = obs["tcp_pose"][3:]
        x, y = r6[:3], r6[3:]
        R_we = np.column_stack([x, y, np.cross(x, y)])
        ff = R_we.T @ info["obj_vel_est"] * dt * self.lead          # 目标速度前馈（末端系）
        a = np.zeros(7)
        a[6] = 1.0
        G, valid = info["observed_grasps_ee"], info["observed_valid"]

        if self.phase in ("approach", "descend"):
            if valid is None or not valid.any():
                self.blind_steps += 1
                a[:3] = ff / c.max_delta_pos_m
                if self.phase == "descend":
                    a[2] += 0.3                                      # 盲区内沿接近方向继续下探
                    if self.blind_steps > 3:
                        self.phase, self.counter = "close", 0
                return np.clip(a, -1, 1)
            self.blind_steps = 0
            g = G[0]
            R = g[:3, :3]
            if rot_angle(R @ RZ_PI) < rot_angle(R):
                R = R @ RZ_PI
            offset = self.pregrasp if self.phase == "approach" else 0.0
            target = g[:3, 3] - offset * R[:, 2]
            err = self.gain * target + ff
            a[:3] = err / c.max_delta_pos_m
            a[3:6] = self.gain * logmap(R) / c.max_delta_rot_rad
            if self.phase == "approach" and np.linalg.norm(target) < 0.012 and rot_angle(R) < 0.12:
                self.phase = "descend"
            elif self.phase == "descend" and np.linalg.norm(g[:3, 3]) < 0.008:
                self.phase, self.counter = "close", 0
            return np.clip(a, -1, 1)

        if self.phase == "close":
            a[:3] = ff / c.max_delta_pos_m
            a[6] = -1.0
            self.counter += 1
            if self.counter >= self.close_steps:
                self.phase = "lift"
            return np.clip(a, -1, 1)

        # lift：世界 +z 在末端系下的方向
        a[:3] = R_we.T @ np.array([0, 0, 1.0])
        a[6] = -1.0
        return np.clip(a, -1, 1)
