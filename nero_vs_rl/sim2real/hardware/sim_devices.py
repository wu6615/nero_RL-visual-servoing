"""MuJoCo stand-ins for the DC1 and NERO reader, for an offline dry run of the calibration pipeline.

``SimRig`` = the project scene + a 12x9 chessboard + the wrist camera at a known
"true" mount.  ``SimDC1Camera`` / ``SimNeroArm`` expose the same methods the
calibration steps use on hardware, so step1 -> step2 -> step3 can be rehearsed
(and tested) before going to the lab; step2 must recover the known mount.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from ... import SCENE_XML
from ...agents.controllers import DiffIK
from ...agents.robots import Nero7Agent
from ...utils.config import EnvConfig
from ..calibration.board import Chessboard
from ..calibration.sim_bridge import cv_cam_to_mujoco
from .dc1_camera import Intrinsics, RGBDFrame

# 仿真里假设的"真实"安装：彩色相机在 gripper_base 侧面、斜看指尖（与 robot.xml 占位接近但不同），
# 深度相机在彩色相机右侧 25 mm。step2 应当把它们准确求回来。
DEFAULT_TRUE_MOUNT = dict(pos=[0.085, 0.004, 0.050], euler_deg=[0.0, -42.0, 2.0])
SIM_T_COLOR_DEPTH = np.array([[1, 0, 0, 0.025], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def mount_T(pos, euler_deg) -> np.ndarray:
    """OpenCV camera pose in gripper_base: optical axis starts along gripper_base +z (approach), then rotate."""
    from ..hardware.nero_arm import euler_xyz_to_R
    base = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])         # 图像 x 沿 gripper_base +y，视线沿 +z
    T = np.eye(4)
    T[:3, :3] = euler_xyz_to_R(*np.radians(euler_deg)) @ base
    T[:3, 3] = pos
    return T


class SimRig:
    def __init__(self, board: Chessboard | None = None, T_base_board: np.ndarray | None = None,
                 X_true: np.ndarray | None = None, color_size=(1280, 720), color_fovy=60.0,
                 depth_size=(640, 400), depth_fovy=58.0, seed: int = 0):
        self.board = board or Chessboard()
        cfg = EnvConfig()
        if T_base_board is None:                     # 板平放在带面上，图案朝上（板 z 轴朝下）
            T_base_board = np.eye(4)
            T_base_board[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
            T_base_board[:3, 3] = [cfg.belt_center_x - 0.10, 0.08, cfg.belt_surface_z + 0.004]
        self.T_base_board = T_base_board
        self.X_true = X_true if X_true is not None else mount_T(**DEFAULT_TRUE_MOUNT)
        self.X_true_depth = self.X_true @ SIM_T_COLOR_DEPTH          # T_grip_depth = T_grip_color · T_color_depth
        self.color_size, self.depth_size = color_size, depth_size
        self.color_fovy, self.depth_fovy = color_fovy, depth_fovy
        self.rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        spec = mujoco.MjSpec.from_file(str(SCENE_XML))
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
        black = spec.add_material(name="calib_black", rgba=[0.03, 0.03, 0.03, 1], specular=0, shininess=0)
        white = spec.add_material(name="calib_white", rgba=[0.92, 0.92, 0.92, 1], specular=0, shininess=0,
                                  emission=0.4)
        R, p = self.T_base_board[:3, :3], self.T_base_board[:3, 3]
        body = spec.worldbody.add_body(name="calib_board", pos=p.tolist(), quat=_quat(R).tolist())
        s, nx, ny = self.board.square_m, self.board.cols + 1, self.board.rows + 1     # 12 x 9 方格
        for a in range(nx):
            for b in range(ny):
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2, s / 2, 0.0005],
                              pos=[(a - 0.5) * s, (b - 0.5) * s, 0.0005], contype=0, conaffinity=0,
                              material="calib_black" if (a + b) % 2 == 0 else "calib_white")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(nx / 2 + 1) * s, (ny / 2 + 1) * s, 0.001],
                      pos=[(nx / 2 - 1) * s, (ny / 2 - 1) * s, 0.0021], contype=0, conaffinity=0,
                      material="calib_white")                           # 白边底板（在图案背后）
        gb = spec.body("gripper_base")
        for name, X, fovy in (("calib_color", self.X_true, self.color_fovy), ("calib_depth", self.X_true_depth, self.depth_fovy)):
            pos, quat = cv_cam_to_mujoco(X)
            gb.add_camera(name=name, pos=pos, quat=quat, fovy=fovy)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.agent = Nero7Agent(self.model)
        self.ik = DiffIK(self.model)
        self._renderers = {}

    def intrinsics(self, which: str) -> Intrinsics:
        (w, h), fovy = (self.color_size, self.color_fovy) if which == "color" else (self.depth_size, self.depth_fovy)
        f = h / 2 / np.tan(np.radians(fovy) / 2)
        return Intrinsics(w, h, f, f, w / 2, h / 2, [0.0] * 5, "sim_pinhole")

    def render(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.color_size if which == "color" else self.depth_size
        if which not in self._renderers:
            self._renderers[which] = mujoco.Renderer(self.model, height=h, width=w)
        r = self._renderers[which]
        mujoco.mj_forward(self.model, self.data)
        r.update_scene(self.data, camera=f"calib_{which}")
        r.disable_depth_rendering()
        rgb = r.render().copy()
        r.enable_depth_rendering()
        depth = r.render().copy()
        r.disable_depth_rendering()
        return rgb, depth

    def set_joints(self, q, gripper_open: float = 0.05):
        """Teleport the arm; gripper fully open (fingers would otherwise hide the board, also on hardware)."""
        self.data.qpos[self.agent.arm_qadr] = q
        g1, g2 = self.agent.grip_qadr
        self.data.qpos[g1], self.data.qpos[g2] = gripper_open, -gripper_open
        mujoco.mj_forward(self.model, self.data)

    def random_view_joints(self, q_seed, n: int, dist=(0.26, 0.40), tilt_deg=35.0, roll_deg=40.0) -> list[np.ndarray]:
        """Joint configs where the color camera looks at the board centre from varied directions."""
        c = self.T_base_board[:3, :3] @ self.board.center + self.T_base_board[:3, 3]
        tcp_in_gb = np.eye(4)
        tcp_in_gb[2, 3] = 0.1325                                      # ee_site 在 gripper_base 前 132.5 mm
        out, tries = [], 0
        while len(out) < n and tries < 40 * n:
            tries += 1
            tilt, az = np.radians(self.rng.uniform(5, tilt_deg)), self.rng.uniform(-np.pi, np.pi)
            d = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])   # 从板指向相机
            cam_p = c + d * self.rng.uniform(*dist)
            z = -d                                                    # 视线指向板中心
            x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            roll = np.radians(self.rng.uniform(-roll_deg, roll_deg))
            Rc = np.column_stack([x, y, z]) @ np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
            T_cam = np.eye(4)
            T_cam[:3, :3], T_cam[:3, 3] = Rc, cam_p
            T_tcp = T_cam @ np.linalg.inv(self.X_true) @ tcp_in_gb
            q, pe, re = self.ik.solve(q_seed, T_tcp[:3, 3], T_tcp[:3, :3], iters=200)
            if pe < 1e-3 and re < 1e-2:
                out.append(q)
        return out


def _quat(R):
    from ...utils.geometry import mat_to_quat_wxyz
    return mat_to_quat_wxyz(R)


class SimDC1Camera:
    """Same surface as DC1Camera (start/stop/read/camera_info), rendered from SimRig."""

    def __init__(self, rig: SimRig):
        self.rig = rig
        self.color_intrinsics = rig.intrinsics("color")
        self.depth_intrinsics = rig.intrinsics("depth")
        self.T_color_depth = SIM_T_COLOR_DEPTH.copy()
        self.device_info = dict(name="SIM DaBai DC1", serial="SIM", pid="0x0", firmware="sim", connection="sim")
        self.active_profiles = dict(color=dict(width=rig.color_size[0], height=rig.color_size[1], fps=30, format="RGB"),
                                    depth=dict(width=rig.depth_size[0], height=rig.depth_size[1], fps=30, format="Y16"))
        self.warnings = ["仿真相机（干跑），不是真机数据"]

    def start(self):
        return self

    def stop(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self, timeout_ms=None, need_color=True) -> RGBDFrame:
        rgb, _ = self.rig.render("color")
        _, depth = self.rig.render("depth")
        return RGBDFrame(rgb, depth.astype(np.float32), time.monotonic())

    def camera_info(self) -> dict:
        return dict(device=self.device_info, profiles=self.active_profiles,
                    color_intrinsics=self.color_intrinsics.to_dict(), depth_intrinsics=self.depth_intrinsics.to_dict(),
                    T_color_depth=self.T_color_depth.tolist(), sdk="sim", warnings=self.warnings,
                    sim_truth=dict(X_color=self.rig.X_true.tolist(), X_depth=self.rig.X_true_depth.tolist(),
                                   T_base_board=self.rig.T_base_board.tolist()))


class SimNeroArm:
    """Same surface as NeroArmReader; ``move_to`` exists only in simulation (auto dry run)."""

    def __init__(self, rig: SimRig, joint_noise_rad: float = 0.0):
        self.rig, self.noise = rig, joint_noise_rad
        self.can_info = dict(channel="sim")

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def close(self):
        pass

    def move_to(self, q):
        self.rig.set_joints(q)

    def joints(self) -> np.ndarray:
        q = self.rig.data.qpos[self.rig.agent.arm_qadr].copy()
        return q + self.rig.rng.normal(0, self.noise, 7) if self.noise else q

    def flange_T(self) -> np.ndarray:
        b = self.rig.model.body("gripper_flange").id
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.rig.data.xmat[b].reshape(3, 3), self.rig.data.xpos[b]
        return T

    def stationary(self, hold_s=1.0, tol_rad=3e-4) -> bool:
        return True
