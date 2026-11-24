"""Put calibration results into the simulation: OpenCV<->MuJoCo camera frames and the site config file.

Yes, the real camera mount goes straight into the sim: the hand-eye result is
T_gripper_base_cam, and the MJCF ``wrist_camera`` is a child of ``gripper_base``,
so it becomes ``env.camera.pos_in_gripper_base / quat_in_gripper_base``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import yaml

from ... import CONFIG_DIR
from ...utils.geometry import mat_to_quat_wxyz, quat_wxyz_to_mat
from .common import SITE_CONFIG

# =============================================================================
# 【说明】相机坐标系约定不同，必须转换：
#   OpenCV（标定得到的）：x 右、y 下、z 朝前（视线方向）
#   MuJoCo camera       ：x 右、y 上、视线沿 -z
#   ⇒ R_body_mj = R_body_cv · diag(1, -1, -1)，位置相同。
# 标定结果写进 algorithms/config/dc1_site.yaml：
#   base: default.yaml   + 只覆盖标定得到的项（相机位姿/内参、盲区、带的位置）
#   provenance:          每一步标定的时间、数据目录、质量指标（只做记录）
# 训练 / 评估默认就用这个文件（nero_vs_rl.TRAIN_CONFIG）。
# =============================================================================

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def cv_cam_to_mujoco(T_body_cam_cv: np.ndarray) -> tuple[list[float], list[float]]:
    R_mj = T_body_cam_cv[:3, :3] @ CV_TO_MJ
    return T_body_cam_cv[:3, 3].tolist(), mat_to_quat_wxyz(R_mj).tolist()


def mujoco_cam_to_cv(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_mat(np.asarray(quat_wxyz, float)) @ CV_TO_MJ
    T[:3, 3] = pos
    return T


def camera_overrides(T_grip_cam_cv: np.ndarray, intr: dict) -> dict:
    """env.camera overrides for a camera with OpenCV extrinsic T_gripper_base_cam and intrinsics dict."""
    pos, quat = cv_cam_to_mujoco(T_grip_cam_cv)
    fovy = float(np.degrees(2 * np.arctan(intr["height"] / 2 / intr["fy"])))
    return dict(pos_in_gripper_base=[round(v, 6) for v in pos], quat_in_gripper_base=[round(v, 8) for v in quat],
                width=int(intr["width"]), height=int(intr["height"]), fovy_deg=round(fovy, 4),
                intrinsics=[round(float(intr[k]), 4) for k in ("fx", "fy", "cx", "cy")])


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def update_site_config(env_updates: dict, provenance: dict, path: Path = SITE_CONFIG) -> Path:
    """Merge env overrides into dc1_site.yaml (creating it with ``base: default.yaml``) and log provenance."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    data = data or {"base": os.path.relpath(CONFIG_DIR / "default.yaml", path.parent)}
    data["env"] = _merge(data.get("env") or {}, env_updates)
    data.setdefault("provenance", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **provenance})
    header = ("# DC1 现场标定结果（由 sim2real/calibration 的 step 脚本自动写入，可手工检查但不要随意改数）。\n"
              "# 训练 / 评估 / 看相机默认就用这个文件（nero_vs_rl.TRAIN_CONFIG）\n")
    path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
