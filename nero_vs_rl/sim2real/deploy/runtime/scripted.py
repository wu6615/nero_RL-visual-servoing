"""Scripted TCP motion through the real execution layer (steps 8-10): straight lines, holds, gripper, full logging."""
from __future__ import annotations

import numpy as np

from ....utils.geometry import expmap, logmap
from .devices import width_to_q

# =============================================================================
# 【说明】现场测量程序（step8 运动响应、step9 夹爪、step10 抓取探针）共用的"脚本动作"：
#   不用策略，但每一拍（10 Hz）走的路径和策略闭环完全一样：
#     期望 TCP 位姿（直线插值）→ IK（单拍关节变化 ≤ 限速 × 0.1 s）→ guard.project（拉回工作空间、
#     指尖抬高、软限位）→ guard.check_target（碰撞、限位）→ streamer.set_target（100 Hz 线性插值发送）
#   每拍还查故障和跟随误差（同 step5）；有问题就原地停，返回问题列表，调用方决定怎么办。
#   记录：
#     rows      每拍一行：动作名、时刻、指令 TCP、实测关节、目标关节、夹爪读数、修正
#     hist      100 Hz 实测关节（真机驱动只缓存最近 5 s，所以每拍都把新样本搬过来）
#     sent      100 Hz 实际发出的关节指令（streamer.sent）
#   指令位姿在 self.p / self.R 里累积，不从实测位姿重新出发（同控制器的做法），所以跟随滞后不会让路线走样。
# =============================================================================


class ScriptedMotion:
    def __init__(self, session):
        self.s = session
        tw = session.twin
        self.ik, self.dt = tw.controller.ik, tw.control_dt
        self.vmax = np.broadcast_to(np.asarray(tw.cfg.max_joint_speed_rad_s, float), (7,))
        self.p, self.R = self.ik.fk(session.streamer.q_cmd)
        self.grip_cmd = session.arm.gripper_width()
        self.rows: list[dict] = []
        self.hist: list[tuple[float, np.ndarray]] = []
        self._last_hist = -np.inf

    # ------------------------------------------------------------ 记录
    def collect(self):
        """Move the arm's newest 100 Hz joint samples into self.hist."""
        arm = self.s.arm
        if hasattr(arm, "history"):
            new = arm.history(self._last_hist)
            if new:
                self.hist += new
                self._last_hist = new[-1][0]

    def measured_tcp(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ik.fk(self.s.arm.joints())

    def _log(self, name: str, now: float, **kw):
        arm = self.s.arm
        self.rows.append(dict(name=name, t=now, p_cmd=self.p.copy(), q=arm.joints(), width=arm.gripper_width(), **kw))
        self.collect()

    # ------------------------------------------------------------ 动作
    def _check(self, now: float) -> list[str]:
        s = self.s
        return s.arm.faults() + s.guard.check_motion(now, s.arm.joints(), s.streamer, executing=s.arm.can_move)

    def _stop(self, problems: list[str], name: str) -> list[str]:
        s = self.s
        s.streamer.hold(s.arm.joints(), s.clock.now())
        self.p, self.R = self.ik.fk(s.streamer.q_cmd)
        self._log(name, s.clock.now(), problems=problems)
        return problems

    def line(self, p_goal, R_goal=None, speed: float = 0.05, name: str = "line", on_tick=None) -> list[str]:
        """Straight TCP line at constant speed (orientation interpolated along). on_tick(k, n, p) may command the gripper."""
        s, dt = self.s, self.dt
        p0, R0 = self.p.copy(), self.R.copy()
        p_goal = np.asarray(p_goal, float)
        R_goal = R0 if R_goal is None else np.asarray(R_goal, float)
        w = logmap(R0.T @ R_goal)
        n = max(1, int(np.ceil(np.linalg.norm(p_goal - p0) / max(speed, 1e-6) / dt)))
        n = max(n, int(np.ceil(np.linalg.norm(w) / 0.15)))                   # 转得多时放慢：每拍最多 0.15 rad
        t0 = s.clock.now()
        for k in range(1, n + 1):
            s.clock.sleep_until(t0 + k * dt)
            now = s.clock.now()
            problems = self._check(now)
            if problems:
                return self._stop(problems, name)
            self.p, self.R = p0 + (p_goal - p0) * k / n, R0 @ expmap(w * k / n)
            if on_tick is not None:
                on_tick(k, n, self.p)
            g = max(width_to_q(s.arm.gripper_width()), width_to_q(self.grip_cmd))   # 夹爪按张得更开的算
            q, _, _ = self.ik.solve(s.streamer.target(), self.p, self.R, iters=10, max_step=self.vmax * dt)
            q, fixes = s.guard.project(q, g)
            problems = s.guard.check_target(s.streamer.target(), q, g)
            if problems:
                return self._stop(problems, name)
            if fixes:                                                        # 被修正：指令位姿跟着改
                self.p, self.R = self.ik.fk(q)
            s.streamer.set_target(q, now, dt)
            self._log(name, now, q_tgt=q, fixes=fixes)
        return []

    def hold(self, seconds: float, name: str = "hold", every: float | None = None) -> list[str]:
        """Keep the current target; log at 10 Hz (or `every` seconds)."""
        s = self.s
        step = every or self.dt
        t0 = s.clock.now()
        for k in range(1, max(1, int(round(seconds / step))) + 1):
            s.clock.sleep_until(t0 + k * step)
            problems = self._check(s.clock.now())
            if problems:
                return self._stop(problems, name)
            self._log(name, s.clock.now())
        return []

    def gripper(self, width_m: float):
        self.grip_cmd = float(width_m)
        self.s.set_gripper(width_m)

    def home(self, q_home: np.ndarray, speed_rad_s: float = 0.4) -> list[str]:
        """Joint-space move back to the start pose (whole path checked first), then re-sync the commanded pose."""
        problems = self.s.move_joints(q_home, speed_rad_s=speed_rad_s)
        self.p, self.R = self.ik.fk(self.s.streamer.q_cmd)
        self.collect()
        return problems

    # ------------------------------------------------------------ 导出
    def arrays(self) -> dict[str, np.ndarray]:
        """100 Hz measured joints and commands (for the .npz next to the report)."""
        sent = list(self.s.streamer.sent)
        return dict(t_meas=np.array([t for t, _ in self.hist]), q_meas=np.array([q for _, q in self.hist]),
                    t_sent=np.array([t for t, _ in sent]), q_sent=np.array([q for _, q in sent]))


def best_lag(t_cmd, x_cmd, t_meas, x_meas, max_lag: float = 0.4) -> tuple[float, float]:
    """Time shift that best explains measured ≈ command(t − lag); returns (lag s, rms at that lag)."""
    keep = (t_meas >= t_cmd[0]) & (t_meas <= t_cmd[-1])                     # 指令最早一拍之前按这一拍算（np.interp 夹边）
    if keep.sum() < 5 or np.ptp(x_cmd) < 1e-4:
        return float("nan"), float("nan")
    lags = np.arange(0.0, max_lag + 1e-9, 0.005)
    rms = [float(np.sqrt(np.mean((x_meas[keep] - np.interp(t_meas[keep] - lag, t_cmd, x_cmd)) ** 2))) for lag in lags]
    i = int(np.argmin(rms))
    return float(lags[i]), rms[i]
