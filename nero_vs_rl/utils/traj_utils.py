"""Planar target trajectories on the belt plane, sampled at the physics rate.

Equivalent of GAP-RL ``utils/traj_utils.py::gen_traj`` but for a conveyor laid
along base Y: ``line`` (belt motion, optional smooth speed variation),
``bezier2d`` (random smooth curve over the belt) and ``static``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import EnvConfig


@dataclass
class Trajectory:
    xy: np.ndarray      # (T, 2) world/base frame
    yaw: np.ndarray     # (T,)
    vel: np.ndarray     # (T, 2) surface velocity (m/s)
    yaw_rate: np.ndarray  # (T,)
    dt: float

    def index(self, t: float) -> int:
        return int(min(max(round(t / self.dt), 0), len(self.yaw) - 1))


def _speed_profile(rng: np.random.Generator, n: int, dt: float, v0: float, vary: bool,
                   lo: float = 0.003, hi: float = np.inf) -> np.ndarray:
    # 【说明】vary_speed 在基准速度上叠加 ±10～35% 的正弦变速；结果裁剪到配置的带速范围内，
    #   保证任何时刻的带速都不超出 trajectory.speed（例如 4～25 cm/s 时不会瞬时冲到 33 cm/s）
    if not vary:
        return np.full(n, v0)
    t = np.arange(n) * dt
    period = rng.uniform(3.0, 8.0)
    amp = rng.uniform(0.1, 0.35)
    phase = rng.uniform(0, 2 * np.pi)
    return np.clip(v0 * (1 + amp * np.sin(2 * np.pi * t / period + phase)), max(lo, 0.003), hi)


RAIL_MARGIN_M = 0.005


def half_width_x_box(hx: float, hy: float) -> Callable[[float], float]:
    """Half extent along base x of a box footprint rotated by yaw (for lateral_full_belt)."""
    return lambda yaw: abs(hx * np.cos(yaw)) + abs(hy * np.sin(yaw))


# 【说明】物体从哪里来：
#   朝向 yaw0 在 [−π, π) 均匀随机（长方体尺寸也随机，所以"哪个面朝上"等效也是随机的）
#   横向 x0：lateral_full_belt=False 时在 belt_center_x + lateral 里均匀取（v5 及以前 ±7 cm，只走带中间）；
#            True 时在整条带宽上均匀取，再按物体在这个朝向下沿 x 的半宽收进护栏内（离护栏 ≥ 5 mm），
#            所以贴着两侧护栏走的物体也会出现（需要 half_width_x：yaw → 沿 x 的半宽，由环境按物体尺寸给出）
#   沿带起点 y0：start_along 里均匀取
def generate(rng: np.random.Generator, cfg: EnvConfig, duration_s: float,
             half_width_x: Callable[[float], float] | None = None) -> Trajectory:
    tc = cfg.trajectory
    dt = cfg.sim_timestep
    n = int(np.ceil(duration_s / dt)) + 2
    sign = float(np.sign(cfg.belt_direction_sign) or -1.0)
    yaw0 = rng.uniform(-np.pi, np.pi)
    if tc.lateral_full_belt:
        room = cfg.belt_half_width - RAIL_MARGIN_M - (half_width_x(yaw0) if half_width_x is not None else 0.0)
        x0 = cfg.belt_center_x + rng.uniform(-max(room, 0.0), max(room, 0.0))
    else:
        x0 = cfg.belt_center_x + rng.uniform(*tc.lateral)
    y0 = rng.uniform(*tc.start_along)
    mode, (lo, hi) = tc.mode, tc.speed
    if tc.static_prob > 0 and rng.random() < tc.static_prob:
        mode = "static"
    elif tc.slow_prob > 0 and rng.random() < tc.slow_prob:
        lo, hi = tc.slow_speed

    if mode == "static":
        xy = np.tile([x0, y0], (n, 1))
        vel = np.zeros((n, 2))
        yaw = np.full(n, yaw0)
        return Trajectory(xy, yaw, vel, np.zeros(n), dt)

    v0 = rng.uniform(lo, hi)
    speed = _speed_profile(rng, n, dt, v0, tc.vary_speed, lo, hi)
    s = np.concatenate([[0.0], np.cumsum(speed[:-1] * dt)])  # 弧长

    if mode == "line":
        xy = np.stack([np.full(n, x0), y0 + sign * s], axis=1)
        vel = np.stack([np.zeros(n), sign * speed], axis=1)
        yaw_rate = np.full(n, rng.uniform(-tc.yaw_rate_max, tc.yaw_rate_max))
        yaw = yaw0 + np.concatenate([[0.0], np.cumsum(yaw_rate[:-1] * dt)])
        return Trajectory(xy, yaw, vel, yaw_rate, dt)

    if mode == "bezier2d":
        half_w = cfg.belt_half_width - 0.05
        length = max(float(s[-1]), 0.05)
        p0 = np.array([x0, y0])
        p3 = np.array([cfg.belt_center_x + rng.uniform(-half_w, half_w), y0 + sign * length])
        p1 = np.array([cfg.belt_center_x + rng.uniform(-half_w, half_w), y0 + sign * length / 3])
        p2 = np.array([cfg.belt_center_x + rng.uniform(-half_w, half_w), y0 + sign * 2 * length / 3])
        u = np.linspace(0, 1, 2000)[:, None]
        curve = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3
        seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        s_scaled = s * arc[-1] / max(s[-1], 1e-9)
        xy = np.stack([np.interp(s_scaled, arc, curve[:, 0]), np.interp(s_scaled, arc, curve[:, 1])], axis=1)
        vel = np.gradient(xy, dt, axis=0)
        heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
        yaw = yaw0 + heading - heading[0]
        yaw_rate = np.gradient(yaw, dt)
        return Trajectory(xy, yaw, vel, yaw_rate, dt)

    raise ValueError(f"未知轨迹模式: {mode}")
