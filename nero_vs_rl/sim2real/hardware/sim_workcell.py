"""The belt cell in MuJoCo behind the same arm / camera / clock interfaces as the lab hardware (closed-loop dry run).

``deploy/runtime`` talks to ``hardware/nero_control.py`` + DC1 in the lab and to this class in a dry run,
so everything from perception to the 100 Hz joint stream runs unchanged before going to the lab.
(``sim_devices.py`` is the calibration stand-in with a chessboard; this one has the belt and objects.)
``CameraEffects`` adds the real DC1's latency, frame-time jitter and depth artefacts (off by default = ideal camera).
"""
from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from ...envs.dynamic_grasp import DynamicGraspEnv
from ...sensors.camera import WristRGBD
from ...utils.config import EnvConfig
from ..deploy.runtime.devices import q_to_width, width_to_q
from .dc1_camera import RGBDFrame
from .nero_arm import interp_joints

# =============================================================================
# 【说明】虚拟实验室 = 一个 DynamicGraspEnv，但不走 env.step()：
#   真机上没有"一步"，只有：100 Hz 收到关节目标 → 机械臂自己跟；相机自己出帧；时间自己流。
#   这里照样拆开：
#     SimArm.send_joints(q)   把 q 写进位置执行器 ctrl（= 真机 move_js），夹爪同理
#     SimCamera.latest()      用腕部相机按当前状态渲染一帧深度（与 step1 --sim 录制相同）
#     SimClock.sleep_until(t) 推进物理到 t：每 10 ms 先调用 every() 注册的回调（100 Hz 发送），
#                             再走 5 个 2 ms 物理子步，子步前后调用环境的钩子（带面、物体驱动、接触）
#   ⇒ 时间是虚拟的：确定性、比实时快；开图形界面时 SimViewer 按实时节奏播放。
#   物体轨迹按 WORLD_TRAJ_S 秒生成（等目标 + 12 s 策略 + 抓起后停住，比训练回合长）。
# =============================================================================
TICK_S = 0.01
WORLD_TRAJ_S = 30.0


# =============================================================================
# 【说明】真机 DC1 有、理想渲染没有的相机效应（默认全关 = 理想相机，同训练）：
#   latency_s     画面是 latency 秒前的场景，帧时间戳却是"现在"（真机时间戳是帧到达电脑的时刻）。
#                 感知按 帧时间戳 − 配置延迟 去取关节算相机位姿；两者相等时补偿正好，不等就是标定误差。
#                 实现：每 10 ms 存一份场景状态（qpos + mocap），渲染时取 t − latency 那一份放进另一个 MjData。
#   latency_jitter_s 每帧实际延迟再加 N(0, jitter)，感知仍按配置延迟补偿（真机曝光时刻不固定）
#   depth_noise_m 0.3 m 处每像素深度噪声标准差，按 (z / 0.3)² 放大（双目深度误差随距离平方增长）
#   edge_dropout  深度跳变 > 1 cm 的边缘像素被置 0（无效）的概率（真机物体边缘常缺深度）
#   2026-11-24 真机实测：延迟 0.08 s，补偿后无系统偏差；静止时逐帧抖动 < 1 mm；机械臂 40～60 cm/s 运动时
#     物体中心逐帧抖 x 6、y 4、z 2 mm（标准差），和速度无线性关系 → 更像帧时刻抖动 / 运动模糊，不是深度噪声。
# =============================================================================
@dataclass
class CameraEffects:
    latency_s: float = 0.0
    latency_jitter_s: float = 0.0
    depth_noise_m: float = 0.0
    edge_dropout: float = 0.0
    seed: int = 0

    @property
    def ideal(self) -> bool:
        return self.latency_s <= 0 and self.latency_jitter_s <= 0 and self.depth_noise_m <= 0 and self.edge_dropout <= 0


def sim_depth_intrinsics(cfg: EnvConfig) -> dict:
    """MuJoCo renders an ideal pinhole: principal point at the image centre, no distortion."""
    c = cfg.camera
    fy = c.intrinsics[1] if c.intrinsics is not None else (c.height / 2) / np.tan(np.deg2rad(c.fovy_deg) / 2)
    return dict(width=c.width, height=c.height, fx=fy, fy=fy, cx=c.width / 2, cy=c.height / 2, dist=[0.0] * 5,
                model="sim_pinhole")


class SimWorkcell:
    def __init__(self, cfg: EnvConfig, viewer: bool = False, playback: float = 1.0,
                 camera_effects: CameraEffects | None = None):
        self.cfg = cfg
        self.effects = camera_effects or CameraEffects()
        self._states: deque = deque(maxlen=100)          # (t, qpos, mocap_pos, mocap_quat)，每 10 ms 一份，留 1 s
        world_cfg = dataclasses.replace(cfg, max_steps=int(WORLD_TRAJ_S * cfg.control_freq),   # 只影响轨迹长度
                                        finger_length_noise_m=0.0)                             # 名义手指尺寸
        self.env = DynamicGraspEnv(world_cfg)
        self.model, self.data, self.agent = self.env.model, self.env.data, self.env.agent
        self.n_sub = int(round(TICK_S / cfg.sim_timestep))
        self.clock = SimClock(self)
        self.arm = SimArm(self)
        self.camera = SimCamera(self)
        self.viewer = None
        if viewer:
            from ...envs.viewer import SimViewer
            self.viewer = SimViewer(self.env, playback)
        self.k = 0

    def reset(self, seed: int, object_types: list[str] | None = None):
        """New episode: arm at a random start pose (as in training), new object and belt trajectory."""
        self.env.reset(seed=seed, options={"object_types": object_types} if object_types else None)
        self.k = 0
        self.arm.reset()
        self._states.clear()
        self.snapshot(self.clock.t)

    def snapshot(self, t: float):
        """Remember the scene at time t (what a delayed camera frame will show later)."""
        d = self.data
        self._states.append((t, d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy()))

    def state_at(self, t: float):
        """Latest remembered scene at or before t (the oldest one if t is earlier than all)."""
        best = self._states[0]
        for st in self._states:
            if st[0] <= t + 1e-9:
                best = st
        return best

    def park_object(self):
        """Take the target out of the scene and stop the belt (motion checks, no-target tests)."""
        self.place_static_object(2.0, 2.0, 0.0)

    def place_static_object(self, x: float, y: float, yaw: float):
        """Put the target at (x, y, yaw) on the belt and keep it (and the belt) still for the whole episode.

        直接把物体和它的驱动载体一起挪过去；只改轨迹的话，weld 约束会在一个子步里把物体拽过去，数值发散。
        """
        env, d = self.env, self.data
        env.traj.xy[:], env.traj.vel[:], env.traj.yaw[:] = [x, y], 0.0, yaw
        env._drive_target(self.k)
        o = env.obj
        d.qpos[o["qadr"]:o["qadr"] + 3] = d.mocap_pos[o["carrier"]]
        d.qpos[o["qadr"] + 3:o["qadr"] + 7] = d.mocap_quat[o["carrier"]]
        d.qvel[o["dadr"]:o["dadr"] + 6] = 0.0
        mujoco.mj_forward(self.model, d)
        self.snapshot(self.clock.t)

    def set_arm(self, q: np.ndarray):
        """Teleport the arm to joints q with the gripper open, and hold it there (start pose of a replayed run)."""
        d, ag = self.data, self.agent
        ag.set_state(d, np.asarray(q, float), self.cfg.gripper_max_q)
        d.ctrl[ag.act_arm] = q
        d.ctrl[ag.act_grip] = self.cfg.gripper_max_q
        mujoco.mj_forward(self.model, d)
        self.arm.reset()
        self.snapshot(self.clock.t)

    def advance(self, dt: float):
        env, d = self.env, self.data
        for _ in range(int(round(dt / self.cfg.sim_timestep))):
            self.k += 1
            env._before_substep(self.k)
            mujoco.mj_step(self.model, d)
            env._after_substep(self.k)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > self.arm.badqacc0:
            self.arm.unstable = True
        self.arm.record(self.clock.t + dt)
        self.snapshot(self.clock.t + dt)

    def truth(self) -> dict:
        """Ground truth the lab does not have: is the object really held and lifted (same rule as env.evaluate)."""
        env, d, c = self.env, self.data, self.cfg
        ct = env._contacts
        grasped = bool(ct["left"] and ct["right"] and self.agent.gripper_opening(d) > 1e-3)
        p = d.xpos[env.obj["body"]]
        lift = float(p[2] - env.obj_half_h - c.belt_surface_z)
        return dict(grasped=grasped, lifted=grasped and lift > c.lift_height_m, lift=lift, obj_type=env.obj_type,
                    obj_pos=p.copy(), fell=bool(p[2] < c.belt_surface_z - 0.05),
                    missed=bool(not grasped and p[1] * np.sign(c.belt_direction_sign) > abs(c.downstream_limit_y)))

    def belt_speed(self) -> float:
        i = min(self.k, len(self.env.traj.vel) - 1)
        return float(np.linalg.norm(self.env.traj.vel[i]))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.camera.close()
        self.env.close()


class SimClock:
    def __init__(self, cell: SimWorkcell):
        self.cell, self.t = cell, 0.0
        self._periodic: list[list] = []                 # [周期, 回调, 下次时刻]

    def now(self) -> float:
        return self.t

    def every(self, period_s: float, fn) -> None:
        self._periodic.append([period_s, fn, self.t])

    def sleep_until(self, t: float) -> None:
        while self.t + 1e-9 < t:
            for item in self._periodic:
                if self.t + 1e-9 >= item[2]:
                    item[1](self.t)
                    item[2] += item[0]
            self.cell.advance(TICK_S)
            self.t = round(self.t + TICK_S, 9)

    def stop(self) -> None:
        self._periodic.clear()


class SimArm:
    can_move = True

    def __init__(self, cell: SimWorkcell):
        self.cell = cell
        self._hist: deque = deque(maxlen=500)
        self.unstable, self.estopped, self.badqacc0 = False, False, 0

    def reset(self):
        self._hist.clear()
        self.unstable, self.estopped = False, False
        self.badqacc0 = self.cell.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        self.record(self.cell.clock.t)

    def record(self, t: float):
        self._hist.append((t, self.joints()))

    def joints(self) -> np.ndarray:
        return self.cell.agent.arm_qpos(self.cell.data)

    def joints_at(self, t: float) -> np.ndarray:
        return interp_joints(list(self._hist), t) if self._hist else self.joints()

    def history(self, since: float = -np.inf):
        return [(t, q.copy()) for t, q in self._hist if t > since]

    def gripper_width(self) -> float:
        return q_to_width(self.cell.agent.gripper_opening(self.cell.data))

    def faults(self) -> list[str]:
        return ["仿真数值发散"] if self.unstable else []

    def send_joints(self, q: np.ndarray) -> None:
        if not self.estopped:
            self.cell.data.ctrl[self.cell.agent.act_arm] = q

    def send_gripper(self, width_m: float) -> None:
        if not self.estopped:
            self.cell.data.ctrl[self.cell.agent.act_grip] = width_to_q(width_m)

    def emergency_stop(self) -> None:
        self.estopped = True
        self.cell.data.ctrl[self.cell.agent.act_arm] = self.joints()


class SimCamera:
    def __init__(self, cell: SimWorkcell):
        self.cell = cell
        self.intrinsics = sim_depth_intrinsics(cell.cfg)
        i = self.intrinsics
        self.K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
        self.D = np.zeros(5)
        self._renderer = None
        self._past = mujoco.MjData(cell.model)
        self._rng = np.random.default_rng(cell.effects.seed)

    def latest(self) -> RGBDFrame:
        cell, fx = self.cell, self.cell.effects
        if self._renderer is None:
            self._renderer = WristRGBD(cell.model, cell.cfg.camera)
        data = cell.data
        lat = fx.latency_s + (self._rng.normal(0.0, fx.latency_jitter_s) if fx.latency_jitter_s > 0 else 0.0)
        if lat > 0:                                              # 画面取 lat 秒前的场景
            _, qpos, mpos, mquat = cell.state_at(cell.clock.t - lat)
            data = self._past
            data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = qpos, mpos, mquat
            mujoco.mj_forward(cell.model, data)
        rgb, depth = self._renderer.render(data)
        if fx.depth_noise_m > 0 or fx.edge_dropout > 0:
            depth = self.degrade(depth)
        return RGBDFrame(rgb, depth, cell.clock.t)               # 时间戳 = 现在（同真机：到达时刻）

    def degrade(self, depth: np.ndarray) -> np.ndarray:
        fx, rng = self.cell.effects, self._rng
        depth = depth.copy()
        valid = depth > 0
        if fx.edge_dropout > 0:
            jump = np.zeros_like(valid)
            jump[:, 1:] |= np.abs(np.diff(depth, axis=1)) > 0.01
            jump[1:, :] |= np.abs(np.diff(depth, axis=0)) > 0.01
            depth[jump & valid & (rng.random(depth.shape) < fx.edge_dropout)] = 0.0
        if fx.depth_noise_m > 0:
            z = depth[depth > 0]
            depth[depth > 0] = z + rng.normal(0.0, 1.0, z.shape) * fx.depth_noise_m * (z / 0.3) ** 2
        return depth.astype(np.float32)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
