"""Batched belt trajectories (line mode): closed-form speed, integrated once per physics step."""
from __future__ import annotations

import math

import torch

from ...utils.config import EnvConfig
from ...utils.traj_utils import RAIL_MARGIN_M
from .objects import ObjectBatch


class BeltTrajectories:
    def __init__(self, cfg: EnvConfig, objects: ObjectBatch, num_envs: int, gen: torch.Generator, device, dtype):
        if cfg.trajectory.mode not in ("line", "static"):
            raise ValueError("Isaac 版只实现了 line / static 轨迹")
        self.cfg, self.obj, self.gen, self.device, self.dtype = cfg, objects, gen, device, dtype
        z = lambda *s: torch.zeros(num_envs, *s, device=device, dtype=dtype)
        self.x0, self.y, self.yaw, self.yaw_rate, self.t = z(), z(), z(), z(), z()
        self.v0, self.amp, self.period, self.phase, self.lo, self.hi = z(), z(), z() + 1, z(), z(), z()
        self.sign = float(math.copysign(1.0, cfg.belt_direction_sign or -1.0))

    def _u(self, n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=self.gen, device=self.device, dtype=self.dtype)

    def reset(self, ids: torch.Tensor):
        c, tc, n = self.cfg, self.cfg.trajectory, len(ids)
        self.t[ids] = 0.0
        self.yaw[ids] = self._u(n, -math.pi, math.pi)
        if tc.lateral_full_belt:
            room = (c.belt_half_width - RAIL_MARGIN_M - self.obj.half_width_x(self.yaw)[ids]).clamp_min(0)
            self.x0[ids] = c.belt_center_x + room * self._u(n, -1, 1)
        else:
            self.x0[ids] = c.belt_center_x + self._u(n, *tc.lateral)
        self.y[ids] = self._u(n, *tc.start_along)
        static = self._u(n, 0, 1) < (1.0 if tc.mode == "static" else tc.static_prob)
        slow = ~static & (self._u(n, 0, 1) < tc.slow_prob)
        lo = torch.where(slow, tc.slow_speed[0], tc.speed[0]).to(self.dtype)
        hi = torch.where(slow, tc.slow_speed[1], tc.speed[1]).to(self.dtype)
        v0 = lo + (hi - lo) * self._u(n, 0, 1)
        self.lo[ids], self.hi[ids] = lo.clamp_min(0.003) * ~static, hi * ~static
        self.v0[ids] = v0 * ~static
        self.amp[ids] = self._u(n, 0.1, 0.35) if tc.vary_speed else 0.0
        self.period[ids], self.phase[ids] = self._u(n, 3.0, 8.0), self._u(n, 0, 2 * math.pi)
        self.yaw_rate[ids] = self._u(n, -tc.yaw_rate_max, tc.yaw_rate_max) * ~static

    def speed(self) -> torch.Tensor:
        v = self.v0 * (1 + self.amp * torch.sin(2 * math.pi * self.t / self.period + self.phase))
        return torch.minimum(torch.maximum(v, self.lo), self.hi)

    def step(self, dt: float):
        self.y += self.sign * self.speed() * dt
        self.yaw += self.yaw_rate * dt
        self.t += dt

    def velocity_y(self) -> torch.Tensor:
        return self.sign * self.speed()

    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(position (B,3) of the object centre, yaw (B,)) in the robot base frame."""
        z = self.cfg.belt_surface_z + self.obj.half[:, 2] + 1e-4
        return torch.stack([self.x0, self.y, z], 1), self.yaw
