"""Shared calibration helpers: MJCF forward kinematics, session folders, acceptance thresholds."""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from ... import PACKAGE_DIR, SCENE_XML, SITE_CONFIG
from ...agents.robots import Nero7Agent

PROJECT_DIR = PACKAGE_DIR.parent
CALIB_DATA_DIR = PROJECT_DIR / "calib_data"          # 现场采集的数据都放这里（每次一个子目录）
# SITE_CONFIG（algorithms/config/dc1_site.yaml）：标定结果 → 仿真/训练配置，定义在包 __init__

# 验收阈值
# 2026-11-22 现场经验：NERO 名义运动学与真实位姿之间有 ~2-3 mm / ~0.5° 的随机误差（关节弹性、背隙等），
# 所以"各样本推算的板位姿偏差"在姿态多样时会超过 5 mm / 1°，这是机械臂本身的绝对精度，多采也降不下来；
# 判断相机外参 X 是否可信改看"去掉一部分数据后 X 变不变"（stability），板位姿偏差只做提示和粗差检查。
MAX_PNP_RMS_PX = 1.0          # 单张图 PnP 重投影 RMS，超过就丢弃该张
MAX_BOARD_SPREAD_M = 0.005    # 提示线：超过说明机械臂绝对精度有限（不判失败）
MAX_BOARD_SPREAD_DEG = 1.0
MIN_SAMPLES = 12
MIN_ROTATION_SPAN_DEG = 30.0  # 样本间最大相对旋转（越大越好，至少 30°）
# X（相机安装位姿）是否可信：去掉一部分数据重算，X 的变化不超过这个量
MAX_X_INSTABILITY_M = 0.003
MAX_X_INSTABILITY_DEG = 0.5
# 单个样本推算的板位姿偏差超过这个量说明有粗差（会话中板被碰、相机松动），而不只是机械臂精度
GROSS_BOARD_SPREAD_M = 0.015
GROSS_BOARD_SPREAD_DEG = 3.0


class RobotFK:
    """Joint angles -> poses of MJCF bodies, using the same model as the simulation."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.agent = Nero7Agent(self.model)

    def body_T(self, q: np.ndarray, body: str = "gripper_base") -> np.ndarray:
        self.data.qpos[self.agent.arm_qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        b = self.model.body(body).id
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[b].reshape(3, 3)
        T[:3, 3] = self.data.xpos[b]
        return T


def new_session_dir(kind: str, root: Path | None = None) -> Path:
    d = (root or CALIB_DATA_DIR) / f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_np_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_np_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []


def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))
