"""One run of the closed loop: build the twin, safety guard and 100 Hz streamer once, then run episodes.

Shared by step4 (MuJoCo cell), step5 (scripted motion check) and steps 6-7 (policy on the lab cell).
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ....utils.config import EnvConfig
from ..common import RobotTwin
from .devices import width_to_q
from .episode import PolicyEpisode
from .safety import SafetyGuard, SafetyProfile
from .streamer import JointStreamer

# =============================================================================
# 【说明】LoopSession 把一次运行里"只建一次"的东西放在一起：
#   twin      运动学孪生（同一份 MuJoCo 模型 + 同一份 IK 控制器）；关节限速按档位 speed_scale 缩放
#   guard     安全检查（会把安全带面、工作空间下限写进孪生）
#   streamer  100 Hz 插值发送（session 存在期间一直在发：没有新目标时重发上一个 = 保持）
#   单点最大变化 = 档位限速 × 10 ms × 1.2，任何情况下关节都不会一步跳远
# 回合之间的动作（回起点、张开夹爪）也走 streamer，和策略控制同一条发送路径。
# =============================================================================
STREAM_PERIOD_S = 0.01


def load_policy(model_path: str) -> Callable[[dict], np.ndarray]:
    """SAC actor on the CPU, deterministic.

    torch 只用 1 个线程：batch=1 的小网络多线程没有好处，反而和 numpy / 相机线程抢 CPU，
    推理时间从中位 10 ms、偶尔 50 ms 以上降到约 2 ms（实测）。
    """
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1)
    model = SAC.load(model_path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def stage_config(cfg: EnvConfig, profile: SafetyProfile) -> EnvConfig:
    vmax = np.broadcast_to(np.asarray(cfg.max_joint_speed_rad_s, float), (7,)) * profile.speed_scale
    return dataclasses.replace(cfg, max_joint_speed_rad_s=vmax.tolist())


class LoopSession:
    def __init__(self, cfg: EnvConfig, profile: SafetyProfile, arm, camera, clock, *, belt_z_safe: float,
                 latency_s: float):
        self.cfg, self.profile = cfg, profile
        self.arm, self.camera, self.clock, self.latency_s = arm, camera, clock, latency_s
        self.twin = RobotTwin(stage_config(cfg, profile))
        self.guard = SafetyGuard(self.twin, profile, belt_z_safe)
        vmax = np.asarray(self.twin.cfg.max_joint_speed_rad_s, float)
        self.streamer = JointStreamer(arm, STREAM_PERIOD_S, max_step_rad=vmax * STREAM_PERIOD_S * 1.2)
        self.streamer.start(arm.joints(), clock)

    def episode(self, policy, **kw) -> PolicyEpisode:
        return PolicyEpisode(self, policy, **kw)

    def move_joints(self, q_goal: np.ndarray, speed_rad_s: float = 0.3) -> list[str]:
        """Joint-space move between episodes (e.g. back to the start pose) through the same guard and streamer.

        The whole straight-line path is checked first; while moving, faults / tracking are checked every 0.1 s.
        Returns the problems (empty = arrived); on a problem the arm stops where it is.
        """
        q_goal = np.asarray(q_goal, float)
        grip_q = width_to_q(self.arm.gripper_width())
        problems = self.guard.check_path(self.streamer.q_cmd, q_goal, grip_q)
        if problems:
            return problems
        T = self.streamer.move_to(q_goal, self.clock.now(), speed_rad_s)
        t_end = self.clock.now() + T + 0.5
        while self.clock.now() < t_end:
            self.clock.sleep_until(self.clock.now() + 0.1)
            problems = self.arm.faults() + self.guard.check_motion(self.clock.now(), self.arm.joints(), self.streamer,
                                                                   executing=self.arm.can_move)
            if problems:
                self.streamer.hold(self.arm.joints(), self.clock.now())
                return problems
        return []

    def set_gripper(self, width_m: float):
        if self.arm.can_move:
            self.arm.send_gripper(width_m)

    def close(self):
        self.streamer.stop()
        self.clock.stop()
        self.twin.close()
