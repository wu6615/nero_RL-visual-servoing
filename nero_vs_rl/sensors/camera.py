"""Wrist (eye-in-hand) camera model: pinhole visibility tests and optional RGB-D rendering.

MuJoCo cameras look along their local -Z with +Y up; image u grows with +X and v
with -Y.  Depth outside [min_depth, max_depth] is invalid, mimicking the DC1
stereo blind zone that hides the target during the final approach; so is the stereo band on the
left edge (u < stereo_band_px_m / depth), which the real depth image never fills.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..utils.config import CameraConfig

# 射线只测试 group 0/1/3：物体、带面和机器人碰撞网格；跳过 group 2 的可视网格（含相机外壳）
RAY_GEOMGROUP = np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8)


def intrinsics(cfg: CameraConfig) -> np.ndarray:
    """3x3 K: calibrated [fx, fy, cx, cy] when given, else an ideal pinhole from fovy."""
    if cfg.intrinsics is not None:
        fx, fy, cx, cy = map(float, cfg.intrinsics)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    f = (cfg.height / 2) / np.tan(np.deg2rad(cfg.fovy_deg) / 2)
    return np.array([[f, 0, cfg.width / 2], [0, f, cfg.height / 2], [0, 0, 1.0]])


def stereo_band_u(depth, cfg: CameraConfig):
    """Left-edge column below which a surface at ``depth`` has no stereo depth (0 when the camera has no band)."""
    return cfg.stereo_band_px_m / np.maximum(depth, 1e-6)


def project(points_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray, cfg: CameraConfig):
    """Return (uv (M,2), depth (M,)) for world points; depth is along the optical axis."""
    pc = (points_world - cam_pos) @ cam_mat          # 相机系坐标 (x 右, y 上, -z 视线)
    depth = -pc[:, 2]
    K = intrinsics(cfg)
    safe = np.where(depth > 1e-6, depth, 1e-6)
    u = K[0, 2] + K[0, 0] * pc[:, 0] / safe
    v = K[1, 2] - K[1, 1] * pc[:, 1] / safe
    return np.stack([u, v], axis=1), depth


def visible_mask(model: mujoco.MjModel, data: mujoco.MjData, cam_id: int, points_world: np.ndarray,
                 target_geom: int, cfg: CameraConfig) -> np.ndarray:
    if len(points_world) == 0:
        return np.zeros(0, dtype=bool)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    uv, depth = project(points_world, cam_pos, cam_mat, cfg)
    m = cfg.image_margin_px
    mask = ((depth >= cfg.min_depth_m) & (depth <= cfg.max_depth_m)
            & (uv[:, 0] >= m) & (uv[:, 0] <= cfg.width - m) & (uv[:, 0] >= stereo_band_u(depth, cfg) + m)
            & (uv[:, 1] >= m) & (uv[:, 1] <= cfg.height - m))
    if cfg.occlusion_check:
        geomid = np.zeros(1, dtype=np.int32)
        for i in np.flatnonzero(mask):
            vec = points_world[i] - cam_pos
            dist = np.linalg.norm(vec)
            hit = mujoco.mj_ray(model, data, cam_pos, vec / dist, RAY_GEOMGROUP, 1, -1, geomid)
            if hit >= 0 and geomid[0] != target_geom and hit < dist - 0.003:
                mask[i] = False
    return mask


class WristRGBD:
    """Offscreen RGB-D renderer for the wrist camera (needs an OpenGL context)."""

    def __init__(self, model: mujoco.MjModel, cfg: CameraConfig, camera: str = "wrist_camera"):
        self.cfg = cfg
        self.camera = camera
        self.renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)

    def render(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.update_scene(data, camera=self.camera)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        depth[(depth < self.cfg.min_depth_m) | (depth > self.cfg.max_depth_m)] = 0.0
        if self.cfg.stereo_band_px_m > 0:                    # 和真机一样，左侧双目无效带没有深度
            u = np.arange(depth.shape[1])[None, :]
            depth[u < stereo_band_u(depth, self.cfg)] = 0.0
        return rgb, depth

    def close(self):
        self.renderer.close()


def depth_to_points(depth: np.ndarray, cfg: CameraConfig) -> np.ndarray:
    """Depth image -> points in the OpenCV camera frame (x right, y down, z forward)."""
    K = intrinsics(cfg)
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
