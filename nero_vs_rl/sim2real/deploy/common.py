"""Deployment helpers: a kinematic twin of the sim robot (TCP / camera poses, controller) and recording I/O."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from ... import REAL_DATA_DIR
from ...envs.dynamic_grasp import DynamicGraspEnv
from ...utils.config import EnvConfig
from ..calibration.sim_bridge import CV_TO_MJ

# =============================================================================
# 【说明】运动学孪生（RobotTwin）：真机只给关节角，其余几何全部用训练时那一份 MuJoCo 模型算：
#   关节角 → TCP 位姿（ee_site）、腕部相机位姿（dc1_site.yaml 里的手眼结果）、可夹持最大宽度、
#   参考夹爪点、EEDeltaPoseController（同一份 IK）。
#   直接构造一个 DynamicGraspEnv 当"孪生"，只用它的模型和控制器，不跑物理、不用它的物体。
# =============================================================================
class RobotTwin:
    def __init__(self, cfg: EnvConfig):
        cfg = dataclasses.replace(cfg, finger_length_noise_m=0.0)      # 孪生用名义手指尺寸
        self.env = DynamicGraspEnv(cfg)
        self.cfg, self.model, self.data = cfg, self.env.model, self.env.data
        self.agent, self.controller = self.env.agent, self.env.controller
        self.max_width, self.N, self.K = self.env.max_width, self.env.N, self.env.K
        self.fixed_pts, self.control_dt = self.env._fixed_pts, self.env.control_dt

    def set_joints(self, q: np.ndarray, gripper_q: float | None = None):
        g = self.cfg.gripper_max_q if gripper_q is None else gripper_q
        self.agent.set_state(self.data, np.asarray(q, float), g)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.agent.tcp_pose(self.data)

    def cam_pose_mj(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.agent.camera
        return self.data.cam_xpos[c].copy(), self.data.cam_xmat[c].reshape(3, 3).copy()

    def T_base_cam_cv(self) -> np.ndarray:
        """Wrist depth camera in OpenCV convention (x right, y down, z forward), base frame."""
        p, R = self.cam_pose_mj()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R @ CV_TO_MJ, p
        return T

    def home_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        return self.env._home_tcp_target()

    def command(self, q: np.ndarray, action: np.ndarray) -> dict:
        """What the controller would send for this action from the measured joints (arm not moved)."""
        self.set_joints(q)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset(self.data)
        p0, R0 = self.tcp_pose()
        q_tgt, grip = self.controller.compute(self.data, action)
        p1, R1 = self.controller.p_cmd, self.controller.R_cmd
        return dict(q_tgt=q_tgt, grip_q=grip, dp_base=p1 - p0, dp_ee=R0.T @ (p1 - p0))

    def close(self):
        self.env.close()


def measured_belt_z(config: str | Path) -> float | None:
    """Belt height measured by calibration step5 with the same wrist camera (provenance in the site config)."""
    import yaml
    data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}   # provenance 只在标定文件里，不经过 base 合并
    for prov in reversed(data.get("provenance") or []):
        if prov.get("step") == "belt" and "measured_surface_z" in prov:
            return float(prov["measured_surface_z"])
    return None


# =============================================================================
# 【说明】录制格式（一个目录一次录制，放数据盘 REAL_DATA_DIR）：
#   meta.json      相机信息（深度内参含畸变）、配置文件、录制频率、来源（real / sim）
#   frames.jsonl   每帧一行：序号、时间戳、关节角、深度/彩色文件名；仿真录制还带物体真值 truth
#   depth/*.png    16 位 PNG，单位 mm（0 = 无效）
#   color/*.jpg    彩色缩到 640 宽，只用于看
# =============================================================================
def new_record_dir(kind: str = "record", root: Path | None = None) -> Path:
    d = Path(root or REAL_DATA_DIR) / f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}"
    (d / "depth").mkdir(parents=True, exist_ok=False)
    (d / "color").mkdir()
    return d


def _np_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def write_json(path: Path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_np_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RecordWriter:
    def __init__(self, out: Path, meta: dict):
        self.out, self.n = Path(out), 0
        write_json(self.out / "meta.json", meta)
        self._f = open(self.out / "frames.jsonl", "a", encoding="utf-8")

    def write(self, stamp: float, joints: np.ndarray, depth: np.ndarray, color: np.ndarray | None = None, **extra):
        i = self.n
        rec = dict(i=i, stamp=float(stamp), joints=np.asarray(joints, float), depth=f"depth/{i:06d}.png", **extra)
        mm = np.clip(np.nan_to_num(depth) * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(self.out / rec["depth"]), mm)
        if color is not None:
            h, w = color.shape[:2]
            small = cv2.resize(color, (640, int(round(h * 640 / w)))) if w > 640 else color
            rec["color"] = f"color/{i:06d}.jpg"
            cv2.imwrite(str(self.out / rec["color"]), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._f.write(json.dumps(rec, ensure_ascii=False, default=_np_default) + "\n")
        self._f.flush()
        self.n += 1

    def close(self):
        self._f.close()


def load_record(d: Path) -> tuple[dict, list[dict]]:
    d = Path(d)
    frames = [json.loads(line) for line in (d / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return read_json(d / "meta.json"), frames


# 【说明】相机延迟补偿：帧的曝光时刻比到达时刻早 latency_s，机械臂在动时必须取那一刻的关节角，
#   否则物体在基座系里的位置会被"拖"一段（TCP 20 cm/s × 70 ms ≈ 14 mm）。
#   有 joints_hist.npz（100 Hz）就在里面插值；老录制没有，就在 10 Hz 的逐帧关节之间插值（粗一些）。
class JointSource:
    def __init__(self, d: Path, frames: list[dict]):
        f = Path(d) / "joints_hist.npz"
        if f.exists():
            z = np.load(f)
            self.t, self.q, self.kind = z["t"], z["q"], "100Hz"
        else:
            self.t = np.array([r["stamp"] for r in frames])
            self.q = np.array([r["joints"] for r in frames], float)
            self.kind = "frames"

    def at(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.t, self.q[:, j]) for j in range(self.q.shape[1])])


def read_depth(d: Path, rec: dict) -> np.ndarray:
    return cv2.imread(str(Path(d) / rec["depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def read_color(d: Path, rec: dict) -> np.ndarray | None:
    if "color" not in rec:
        return None
    img = cv2.imread(str(Path(d) / rec["color"]))
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
