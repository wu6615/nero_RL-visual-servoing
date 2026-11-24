"""Analytic parallel-jaw grasps for box/cylinder targets and GAP-RL style gripper point sets.

Grasp pose convention == desired ``ee_site`` (TCP) pose:
  TCP +z = approach direction (into the object), TCP y = finger closing axis,
  TCP x = y × z.  The TCP sits 10 mm behind the fingertip plane (robot.xml).
Grasps are expressed in the target body frame (z up while resting on the belt).
Stand-in for GAP-RL's LoG detector until a learned grasp detector is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.geometry import rot_axis

TIP_AHEAD_OF_TCP = 0.010     # 指尖平面在 TCP 前 10 mm
MAX_FINGER_DEPTH = 0.035     # 指尖最多伸入物体顶面以下
BELT_CLEARANCE = 0.004       # 指尖离带面最少 4 mm
FINGER_HALF_OPEN = 0.05      # 夹爪半开口（点采样尺度，对应 GAP-RL gripper_w）
FINGER_LENGTH = 0.073


@dataclass
class GraspSet:
    T_obj: np.ndarray    # (M, 4, 4)
    vis_obj: np.ndarray  # (M, 3) 用于可见性判断的表面点（物体系）
    scores: np.ndarray   # (M,)

    def __len__(self) -> int:
        return len(self.scores)


def _grasp_frame(close_dir: np.ndarray, tilt: float) -> np.ndarray:
    c = close_dir / np.linalg.norm(close_dir)
    a = rot_axis(c, tilt) @ np.array([0.0, 0.0, -1.0])
    b = np.cross(c, a)
    return np.stack([b, c, a], axis=1)


def _append(out, R, tip_xyz, top_z, score):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tip_xyz - TIP_AHEAD_OF_TCP * R[:, 2]
    out[0].append(T)
    out[1].append(np.array([tip_xyz[0], tip_xyz[1], top_z]))
    out[2].append(score)


def box_grasps(half: np.ndarray, max_width: float, max_tilt_deg: float) -> GraspSet:
    hx, hy, hz = map(float, half)
    tip_z = max(-hz + BELT_CLEARANCE, hz - MAX_FINGER_DEPTH)
    tilts = np.deg2rad([0.0, -max_tilt_deg, max_tilt_deg]) if max_tilt_deg > 0 else [0.0]
    out = ([], [], [])
    for axis, (h_close, h_along) in enumerate(((hx, hy), (hy, hx))):
        if 2 * h_close > max_width:
            continue
        close_dir = np.eye(3)[axis]
        along_axis = 1 - axis
        n_along = 3 if h_along > 0.025 else 1
        for off in np.linspace(-(h_along - 0.012), h_along - 0.012, n_along) if n_along > 1 else [0.0]:
            for tilt in tilts:
                tip = np.zeros(3)
                tip[along_axis] = off
                tip[2] = tip_z
                _append(out, _grasp_frame(close_dir, tilt), tip, hz, 1.0 - 0.2 * abs(tilt) / max(np.deg2rad(max_tilt_deg), 1e-6))
    return _pack(out)


def cylinder_grasps(radius: float, half_h: float, max_width: float, max_tilt_deg: float) -> GraspSet:
    out = ([], [], [])
    if 2 * radius <= max_width:
        tip_z = max(-half_h + BELT_CLEARANCE, half_h - MAX_FINGER_DEPTH)
        for yaw in np.deg2rad(np.arange(0, 180, 30)):
            close_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            for tilt in ([0.0, np.deg2rad(max_tilt_deg)] if max_tilt_deg > 0 else [0.0]):
                _append(out, _grasp_frame(close_dir, tilt), np.array([0, 0, tip_z]), half_h, 1.0 if tilt == 0 else 0.8)
    return _pack(out)


def _pack(out) -> GraspSet:
    if not out[0]:
        return GraspSet(np.zeros((0, 4, 4)), np.zeros((0, 3)), np.zeros(0))
    return GraspSet(np.array(out[0]), np.array(out[1]), np.array(out[2], dtype=float))


def gripper_keypoints(k: int) -> np.ndarray:
    """Fixed outline of the NERO two-finger gripper in the TCP frame (k points)."""
    w, tip = FINGER_HALF_OPEN, TIP_AHEAD_OF_TCP
    back = tip - FINGER_LENGTH
    per = max(k // 4, 1)
    left = np.stack([np.zeros(per), np.full(per, w), np.linspace(tip, back, per)], 1)
    right = left * np.array([1, -1, 1])
    palm = np.stack([np.zeros(per), np.linspace(-w, w, per), np.full(per, back)], 1)
    stem = np.stack([np.zeros(k - 3 * per), np.zeros(k - 3 * per), np.linspace(back, back - 0.04, k - 3 * per)], 1)
    return np.concatenate([left, right, palm, stem])[:k]


def sample_gripper_points(mode: str, k: int, rng: np.random.Generator, fixed: np.ndarray | None = None) -> np.ndarray:
    if mode == "gauss":
        return rng.normal(0.0, FINGER_HALF_OPEN / 3, size=(k, 3))
    if mode in ("gauss_fix", "keypoints"):
        assert fixed is not None
        return fixed
    raise ValueError(f"未知 points_mode: {mode}")
