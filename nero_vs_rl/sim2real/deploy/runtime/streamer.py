"""100 Hz joint target stream: linear interpolation from the current command to the latest 10 Hz policy target.

Counterpart of the NERO project's ``runtime/joint_stream_sender.py`` (read-only reference) and of
``envs/base_env.py::_simulate``, which interpolates the same way inside one control step.
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

# =============================================================================
# 【说明】为什么要 100 Hz 插值发送：
#   策略 10 Hz 出一个关节目标，两个目标之间最多差 3.9 rad/s × 0.1 s ≈ 0.39 rad。直接发给 move_js
#   （SDK 标注"无平滑、尽快到达"）会让关节猛跳。仿真训练时也不是一步跳过去的：base_env._simulate 在
#   50 个物理子步里线性插值。真机照做——每 10 ms 发一个插值点，单点变化 ≤ 0.04 rad，
#   和 NERO 工程现场验证过的做法一致（joint_command_mode: js，control_hz: 100）。
#
# 一段"插值段" = (起点 q0, 终点 q1, 开始时刻 t0, 时长 T)：
#   set_target(q1)  从"当前已发出的指令"出发（不是实测值，避免和位置环滞后打架），T = 一个控制周期
#   tick(now)       w = (now − t0)/T 截到 [0, 1]，发 q0 + w(q1 − q0)；到终点后一直重发终点 = 保持
#   hold(q)         把目标换成 q（通常是实测关节）= 原地停住
#   move_to(q, v)   关节空间匀速走到 q（回起点用），时长按最慢的关节和限速 v 算
# 每个发送点再按"单点最大变化 max_step"截一次：任何上游错误都不可能让关节一步跳很远。
# =============================================================================
class JointStreamer:
    def __init__(self, arm, period_s: float, max_step_rad: np.ndarray):
        self.arm, self.period = arm, period_s
        self.max_step = np.asarray(max_step_rad, float)
        self._lock = threading.Lock()
        self.q_cmd: np.ndarray | None = None          # 最近一次发出（或预演时"本应发出"）的关节指令
        self._seg = None
        self.n_sent, self.n_clamped = 0, 0
        self.sent: deque = deque(maxlen=6000)           # 最近 60 s 发出的 (时刻, 关节)，step5 用来算跟随延迟
        self.last_tick: float | None = None
        self.fault: str | None = None
        self._stopped = False

    # ------------------------------------------------------------------ 生命周期
    def start(self, q_now: np.ndarray, clock):
        """Begin streaming a hold at ``q_now``; ``clock.every`` then calls ``tick`` at 100 Hz."""
        q = np.asarray(q_now, float).copy()
        with self._lock:
            self.q_cmd = q
            self._seg = (q, q, clock.now(), 0.0)
        clock.every(self.period, self.tick)

    def reset_to(self, q: np.ndarray, now: float):
        """Jump the command to ``q`` without interpolating — only for the MuJoCo cell, whose reset teleports the arm."""
        q = np.asarray(q, float).copy()
        with self._lock:
            self.q_cmd, self._seg = q, (q, q, now, 0.0)

    def stop(self):
        """No further sends.  The arm keeps the last position target it received (position control)."""
        with self._lock:
            self._stopped = True

    # ------------------------------------------------------------------ 设目标
    def set_target(self, q_target: np.ndarray, now: float, duration_s: float):
        with self._lock:
            self._seg = (self.q_cmd.copy(), np.asarray(q_target, float).copy(), now, max(duration_s, 1e-6))

    def hold(self, q: np.ndarray, now: float):
        self.set_target(q, now, self.period)

    def move_to(self, q_goal: np.ndarray, now: float, speed_rad_s: float) -> float:
        """Joint-space move at ``speed_rad_s`` on the slowest joint; returns the duration (s)."""
        dq = np.abs(np.asarray(q_goal, float) - self.q_cmd)
        T = float(dq.max() / speed_rad_s) if dq.max() > 0 else self.period
        self.set_target(q_goal, now, T)
        return T

    def target(self) -> np.ndarray:
        with self._lock:
            return self._seg[1].copy()

    def done(self, now: float) -> bool:
        with self._lock:
            q0, q1, t0, T = self._seg
        return now >= t0 + T

    # ------------------------------------------------------------------ 100 Hz
    def tick(self, now: float):
        with self._lock:
            if self._stopped or self.fault is not None or self._seg is None:
                return
            q0, q1, t0, T = self._seg
            w = 1.0 if T <= 0 else float(np.clip((now - t0) / T, 0.0, 1.0))
            q = q0 + w * (q1 - q0)
            step = q - self.q_cmd
            if np.any(np.abs(step) > self.max_step):
                q = self.q_cmd + np.clip(step, -self.max_step, self.max_step)
                self.n_clamped += 1
            try:
                if self.arm.can_move:
                    self.arm.send_joints(q)
            except Exception as e:                     # 发送失败：停止发送，交给控制循环处理（急停）
                self.fault = f"关节指令发送失败：{type(e).__name__}: {e}"
                return
            self.q_cmd = q
            self.n_sent += 1
            self.last_tick = now
            self.sent.append((now, q))
