"""One closed-loop grasp episode on the belt: wait for a target -> policy servo at 10 Hz -> stop after the lift.

Counterpart of the main loop in GAP-RL ``sim2real/sim2real_traj_following.py``.  Each tick runs the same steps
as ``envs/dynamic_grasp.py`` does in simulation (perception -> memory -> observation -> policy -> controller),
with the simulated camera replaced by DC1 depth and the physics replaced by the arm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from ....envs.observation import GraspTracker, assemble_obs, gripper_points
from ....utils.geometry import make_T
from ..perception import BeltObjectDetector, Detection, observe
from .devices import q_to_width, width_to_q

# =============================================================================
# 【说明】一个回合的状态机：
#
#   WAIT ──第一次"完整检出"物体──→ SERVO ──抓住并抬高 lift_height──→ HOLD ──停 hold_s 秒──→ DONE
#    │                              │ └──策略走满 max_steps 拍（12 s，同训练）─────────────→ DONE
#    └──等了 wait_s 秒还没有──→ DONE│
#                                   └──安全检查不过 / 操作者停止 ──→ ABORT（原地停住）
#   任何阶段：机械臂报故障（驱动报错、未使能、通信中断）──→ ABORT（急停）
#
#   WAIT  机械臂停在起点不动；感知和记忆照常跑（测速窗口提前攒数据）。
#         为什么要等：训练里"完全没有抓取信息"几乎只出现在已经抓住、物体进了盲区的时候，
#         策略把它学成了"闭合夹爪 + 后退"（deploy/README 已知差异 3）——所以看到物体之前不能让策略动。
#         "完整检出" = 物体轮廓没被画面边缘截断、有可见抓取（和策略看到的观测同一个判据）。
#   SERVO 每拍：观测 → 策略 → 本档位处理（限速、锁夹爪）→ 控制器（同一份 IK）→ 安全修正与检查（safety.py）→ 100 Hz 发送
#   HOLD  抓起后停在原地，夹爪保持闭合（hold_after_lift=False 时不进 HOLD，策略一直跑到 max_steps）
#
# 每一拍（10 Hz）的顺序：
#   ① 读传感器：最新深度帧、当前关节、夹爪开口
#   ② 查故障（急停）和运行状态（原地停）
#   ③ 感知：按"帧曝光时刻"的关节角算相机位姿（相机延迟补偿）→ 检出物体 → 抓取 → 记忆/测速
#   ④ 按状态机走一步
#   ⑤ 记录、回调（预览窗口 / 仿真图形界面 / 操作者按键停止）
# =============================================================================
class Phase(str, Enum):
    WAIT = "等目标"
    SERVO = "策略伺服"
    HOLD = "抓起后停住"
    DONE = "结束"
    ABORT = "中止"


START_AFTER_USABLE_FRAMES = 1     # 第一次完整检出就启动（训练回合也是物体一出现策略就在动）
GRIPPER_RESEND_M = 0.002          # 夹爪目标变化超过 2 mm 才重发


@dataclass
class EpisodeResult:
    outcome: str                  # lifted 抓起 | timeout 走满 max_steps 没抓起 | no_target 没等到物体 | stopped 操作者/安全停 | estop 急停
    policy_steps: int = 0
    reasons: list[str] = field(default_factory=list)
    t_start_servo: float | None = None
    t_lifted: float | None = None
    rows: list[dict] = field(default_factory=list)


# =============================================================================
# 【说明】在线感知：和 step2_replay 的离线回放是同一串调用，只是帧来自相机线程而不是录制文件。
#   没有新帧（相机比 10 Hz 慢、或卡了一下）→ 当作这一拍没看到，记忆年龄 +1，和仿真的"看不见"一样处理。
# =============================================================================
@dataclass
class Seen:
    grasps: np.ndarray            # 进观测的抓取（基座系，可能是记忆外推的）
    scores: np.ndarray
    visible: bool                 # 这一拍真的看到了
    usable: bool                  # 这一帧完整检出（没截断、有可见抓取）
    det: Detection | None
    why: str
    frame_stamp: float | None


class LivePerception:
    def __init__(self, cfg, twin, K: np.ndarray, D: np.ndarray, latency_s: float):
        self.cfg, self.twin, self.latency = cfg, twin, latency_s
        self.detector = BeltObjectDetector(cfg, K, D, twin.max_width)
        self.tracker = GraspTracker(cfg.grasp_obs, twin.control_dt)
        self._last_stamp = None

    def reset(self):
        self.detector.reset()
        self.tracker.reset()
        self._last_stamp = None

    def update(self, frame, arm, t: float) -> Seen:
        if frame is None or frame.stamp == self._last_stamp:
            G, s, vis = self.tracker.update(t, None)
            return Seen(G, s, vis, False, None, "没有新帧", None if frame is None else frame.stamp)
        self._last_stamp = frame.stamp
        tw = self.twin
        tw.set_joints(arm.joints_at(frame.stamp - self.latency))          # 曝光时刻的关节 → 相机位姿
        p, R = tw.tcp_pose()
        cam_p, cam_R = tw.cam_pose_mj()
        det, Gw, sc, why = observe(self.detector, frame.depth, tw.T_base_cam_cv(), make_T(R, p), cam_p, cam_R, self.cfg)
        if self.detector.new_object:                                      # 换了物体：记忆和测速窗口重来
            self.tracker.reset()
        G, s, vis = self.tracker.update(t, Gw, sc, det.center if Gw is not None else None)
        return Seen(G, s, vis, Gw is not None, det, why, frame.stamp)


# =============================================================================
# 【说明】只靠真机也有的测量判断"抓住了、抬起来了"（仿真真值只在干跑报告里用）：
#   抓住 = 指令要闭合，但实测开口比指令大 ≥ 5 mm（被物体撑住）、开口 ≥ 1.5 cm（不是夹空），且连续 2 拍不再变化
#   抬起 = 抓住以后 TCP 比抓住那一刻高了 lift_height_m（6 cm，同仿真的成功判据）
# =============================================================================
class GraspMonitor:
    MIN_WIDTH_M, GAP_M, STABLE_M, TICKS = 0.015, 0.005, 0.001, 2

    def __init__(self, lift_height_m: float):
        self.lift_height = lift_height_m
        self.reset()

    def reset(self):
        self.n, self.z_grasp, self._w_prev = 0, None, None

    def update(self, cmd_width: float, meas_width: float, tcp_z: float) -> tuple[bool, bool]:
        stable = self._w_prev is not None and abs(meas_width - self._w_prev) < self.STABLE_M
        self._w_prev = meas_width
        blocked = meas_width > cmd_width + self.GAP_M and meas_width > self.MIN_WIDTH_M
        self.n = self.n + 1 if (blocked and stable) else 0
        held = self.n >= self.TICKS
        if not held:
            self.z_grasp = None
        elif self.z_grasp is None:
            self.z_grasp = tcp_z
        return held, bool(held and tcp_z - self.z_grasp >= self.lift_height)


class PolicyEpisode:
    def __init__(self, session, policy: Callable[[dict], np.ndarray], *, wait_s: float = 30.0, hold_s: float = 2.0,
                 max_steps: int | None = None, on_tick: Callable[["PolicyEpisode", dict], bool] | None = None,
                 log=None, episode_index: int = 0, seed: int = 0):
        s = self.session = session
        self.cfg, self.twin, self.arm, self.camera = s.cfg, s.twin, s.arm, s.camera
        self.clock, self.streamer, self.guard, self.profile = s.clock, s.streamer, s.guard, s.profile
        self.policy, self.on_tick, self.log, self.index = policy, on_tick, log, episode_index
        self.dt = self.twin.control_dt
        self.wait_s, self.hold_s = wait_s, hold_s
        self.max_steps = max_steps or self.cfg.max_steps
        self.perception = LivePerception(self.cfg, self.twin, self.camera.K, self.camera.D, s.latency_s)
        self.monitor = GraspMonitor(self.cfg.lift_height_m)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ 主循环
    def run(self) -> EpisodeResult:
        self._begin()
        k = 0
        while self.phase not in (Phase.DONE, Phase.ABORT):
            k += 1
            target = self.t0 + k * self.dt
            try:
                self.clock.sleep_until(target)
                now = self.clock.now()
                if now - target > self.dt:                       # 上一拍算太久：跳过过期的节拍，不补
                    self.overruns += 1
                    k = int((now - self.t0) / self.dt)
                self._tick(now)
            except KeyboardInterrupt:
                self._stop_here(["操作者按了 Ctrl+C"], "stopped")
            except Exception as e:                               # 程序错误：原地停住再抛出
                self._stop_here([f"程序异常：{type(e).__name__}: {e}"], "stopped")
                raise
        return self.result

    def _begin(self):
        self.t0 = self.clock.now()
        self.phase, self.steps, self.overruns = Phase.WAIT, 0, 0
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.result = EpisodeResult("no_target")
        self.q_start = self.arm.joints()
        self.streamer.hold(self.q_start, self.t0)               # 起点原地保持
        self._grip_sent, self._usable, self._t_hold, self.seen = None, 0, None, None
        self.perception.reset()
        self.monitor.reset()

    def _tick(self, now: float):
        t_compute = time.perf_counter()
        t = now - self.t0
        # ① 读传感器
        q_now, width = self.arm.joints(), self.arm.gripper_width()
        frame = self.camera.latest()
        row = dict(ep=self.index, t=round(t, 3), phase=self.phase.name, q=q_now, width=width)
        # ② 故障 → 急停；运行状态不对 → 原地停
        faults = self.arm.faults()
        if faults:
            return self._finish_row(row, self._emergency(faults), t_compute)
        problems = self.guard.check_motion(now, q_now, self.streamer, executing=self.arm.can_move)
        if self.phase != Phase.HOLD:                             # 停住阶段不再需要相机
            problems += self.guard.check_camera(now, None if frame is None else frame.stamp)
        if problems:
            return self._finish_row(row, self._stop_here(problems, "stopped", q_hold=q_now), t_compute)
        # ③ 感知
        seen = self.seen = self.perception.update(frame, self.arm, t)
        row.update(visible=seen.visible, usable=seen.usable, why=seen.why, mem_age=int(min(self.perception.tracker.mem_age, 999)))
        if seen.det is not None:
            row.update(obj_type=seen.det.obj_type, center=seen.det.center, half=seen.det.half)
        # ④ 状态机
        if self.phase == Phase.WAIT:
            self._wait(seen, now, row)
        elif self.phase == Phase.SERVO:
            self._servo(seen, q_now, width, now, row)
        elif self.phase == Phase.HOLD and now - self._t_hold >= self.hold_s:
            self.phase = Phase.DONE
        self._finish_row(row, None, t_compute, frame)

    # ------------------------------------------------------------------ 各阶段
    def _wait(self, seen: Seen, now: float, row: dict):
        self._usable = self._usable + 1 if seen.usable else 0
        if self._usable >= START_AFTER_USABLE_FRAMES:
            self.phase = Phase.SERVO
            self.result.t_start_servo = now - self.t0
            self.twin.set_joints(self.arm.joints(), width_to_q(self.arm.gripper_width()))
            self.twin.controller.reset(self.twin.data)            # 指令位姿从实测出发（同 env.reset）
        elif now - self.t0 > self.wait_s:
            self.phase = Phase.DONE
            self.result.reasons.append(f"{self.wait_s:.0f} s 内没有完整检出物体")

    def _servo(self, seen: Seen, q_now: np.ndarray, width: float, now: float, row: dict):
        tw, prof = self.twin, self.profile
        tw.set_joints(q_now, width_to_q(width))
        p, R = tw.tcp_pose()
        obs = self._observation(seen, p, R, width)
        a_raw = np.asarray(self.policy(obs), dtype=np.float32)
        a = a_raw.astype(np.float64).copy()
        a[:6] *= prof.speed_scale                                # 本档位限速（关节限速在孪生配置里已同比缩放）
        if not prof.allow_close_gripper:
            a[6] = 1.0                                           # 跟随阶段：夹爪锁定全开
        q_ctrl, grip_q = tw.controller.compute(tw.data, a)
        g_safe = max(width_to_q(width), grip_q)                   # 夹爪按张得更开的算（倾斜时指尖更低）
        problems = self.guard.check_step(self.streamer.target(), tw.controller.last_ik_q)   # ① IK 结果（修正前）不超限速
        q_tgt, fixes = self.guard.project(q_ctrl, g_safe)                     # ② 修正进安全范围
        problems += self.guard.check_pose(q_tgt, g_safe)                      # ③ 修正后的目标必须安全
        if fixes:                                                # 目标被修正：控制器的指令位姿跟着改，下一拍从这里累积
            ctl = tw.controller
            ctl.q_cmd = q_tgt
            ctl.p_cmd, ctl.R_cmd = ctl.ik.fk(q_tgt)
        if tw.controller.last_fixes or fixes:                    # 控制器自己（按配置带面）+ 安全层（按安全带面）的修正
            row["fixes"] = tw.controller.last_fixes + fixes
        row.update(action=a_raw, q_tgt=q_tgt, grip_cmd=q_to_width(grip_q), tcp=p)
        if problems:
            row["problems"] = self._stop_here(problems, "stopped", q_hold=q_now)
            return
        self.streamer.set_target(q_tgt, now, self.dt)
        self._send_gripper(q_to_width(grip_q))
        self.prev_action = a_raw                                 # 下一拍的 obs["action"] = 策略自己的输出（同训练）
        self.steps += 1
        held, lifted = self.monitor.update(q_to_width(grip_q), width, p[2])
        row.update(held=held, lifted=lifted, step=self.steps)
        if lifted:
            self.result.outcome, self.result.t_lifted = "lifted", now - self.t0
            if prof.hold_after_lift:
                self.phase, self._t_hold = Phase.HOLD, now
                self.streamer.hold(self.streamer.q_cmd, now)    # 停在当前指令位置，夹爪保持
                return
        if self.steps >= self.max_steps:
            self.phase = Phase.DONE
            if self.result.outcome != "lifted":
                self.result.outcome = "timeout"
            self.streamer.hold(self.streamer.q_cmd, now)

    def _observation(self, seen: Seen, p: np.ndarray, R: np.ndarray, width: float) -> dict:
        tw, g = self.twin, self.cfg.grasp_obs
        pts = gripper_points(g, tw.K, self.rng, tw.fixed_pts)
        obs, _, _ = assemble_obs(p, R, seen.grasps, seen.scores, seen.visible, self.perception.tracker.age_norm,
                                 width_to_q(width) / self.cfg.gripper_max_q, self.prev_action, pts, tw.N)
        return obs

    def _send_gripper(self, width_m: float):
        if self._grip_sent is None or abs(width_m - self._grip_sent) > GRIPPER_RESEND_M:
            if self.arm.can_move:
                self.arm.send_gripper(width_m)
            self._grip_sent = width_m

    # ------------------------------------------------------------------ 停止
    def stop_requested(self, reason: str = "操作者停止"):
        """Called from on_tick (preview key) to end the episode where it stands."""
        self._stop_here([reason], "stopped", q_hold=self.arm.joints())

    def _stop_here(self, reasons: list[str], outcome: str, q_hold: np.ndarray | None = None) -> list[str]:
        q = self.arm.joints() if q_hold is None else q_hold
        self.streamer.hold(q, self.clock.now())                 # 原地停：目标换成实测关节
        self.phase = Phase.ABORT
        if self.result.outcome != "lifted":
            self.result.outcome = outcome
        self.result.reasons += reasons
        return reasons

    def _emergency(self, faults: list[str]) -> list[str]:
        self.streamer.stop()
        if self.arm.can_move:
            self.arm.emergency_stop()
        self.phase = Phase.ABORT
        self.result.outcome = "estop"
        self.result.reasons += faults
        return faults

    def _finish_row(self, row: dict, problems: list[str] | None, t_compute: float, frame=None):
        row["compute_ms"] = round((time.perf_counter() - t_compute) * 1000, 1)
        row["q_cmd"] = None if self.streamer.q_cmd is None else self.streamer.q_cmd.copy()
        if problems:
            row["problems"] = problems
        row["phase_after"] = self.phase.name
        self.result.policy_steps = self.steps
        self.result.rows.append(row)
        if self.log is not None:
            self.log.tick(row, frame, self.arm)
        if self.on_tick is not None and self.phase not in (Phase.DONE, Phase.ABORT):
            if self.on_tick(self, row):
                self.stop_requested()
