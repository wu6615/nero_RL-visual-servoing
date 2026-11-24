"""Safety profiles per lab stage and the per-tick checks every joint target must pass before it is streamed.

The NERO project runs a scene-collision monitor next to its 100 Hz sender (``runtime_sweep``); here the
same idea is done synchronously on the kinematic twin, once per 10 Hz target.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# =============================================================================
# 【说明】安全档位（SafetyProfile）：每个现场阶段一组限值，step 脚本按阶段选，不在命令行上零散地调。
#   speed_scale          策略动作的平移/旋转 × 此值，关节限速 × 此值（训练时是 1.0）
#                        ★ 抓取不能降速：策略按训练时的速度掐好了"一边下降一边闭合"的时机，降速后手还没到就闭合了，
#                          夹空再重试直到超时（仿真干跑 100 回合：全速 76%，×0.5 只有 36%，带速 < 8 cm/s 也只有 54%）。
#                          所以只有不抓的跟随档降速；抓取档的安全靠下面几项 + 慢带/静止物体起步 + 操作者拍急停
#   tcp_floor_above_belt TCP 最低高度，相对"安全带面"（配置值和标定实测值里更高的那个，见 SafetyGuard）
#   finger_clearance_m   夹爪（指尖、夹爪座）最低点至少高出安全带面这么多；不够就把目标整体抬高（见 project）
#                        None = 和训练一样不管（仿真干跑）。为什么不只靠 TCP 下限：TCP 在指尖平面后 10 mm，
#                        但夹爪张开时绕闭合方向倾斜 15°，外侧指尖就比 TCP 低 24 mm；训练里策略常把指尖顶到带面上
#                        （仿真里带面把手指挡住，不扣分），真机不能这样
#   allow_close_gripper  False ⇒ 夹爪始终张开（跟随阶段）
#   hold_after_lift      抓住并抬高 lift_height 后停在原地，不让策略带着物体继续走
#                        （仿真里 v5 抬起后会把物体快速移到工作空间角上，真机不需要这一段）
#   max_tracking_err_rad 实测关节和已发指令相差超过它 ⇒ 停（被挡住、撞到东西、或机械臂没在跟）
#                        跟随误差 ≈ 关节速度 × 位置环滞后，所以随速度放宽：0.15 + 0.4 × speed_scale rad。
#                        依据：仿真位置环滞后约 0.2 s，全速时每回合最大误差中位 0.26、最大 0.43 rad；
#                        0.5 倍速时最大 0.22 rad。真机滞后由 step5 测，比仿真小很多时可以收紧。
#                        全速时被挡住和正常滞后分不太开，碰撞主要靠驱动器自己的碰撞报警（nero_control.faults）
#   max_frame_age_s      相机这么久没有新帧 ⇒ 停
# =============================================================================
TRACK_ERR_BASE_RAD = 0.15
TRACK_ERR_PER_SPEED_RAD = 0.40


@dataclass(frozen=True)
class SafetyProfile:
    name: str
    title: str
    speed_scale: float
    tcp_floor_above_belt_m: float
    finger_clearance_m: float | None
    allow_close_gripper: bool
    hold_after_lift: bool = True
    max_frame_age_s: float = 0.5

    @property
    def max_tracking_err_rad(self) -> float:
        return TRACK_ERR_BASE_RAD + TRACK_ERR_PER_SPEED_RAD * self.speed_scale


PROFILES = {
    "sim": SafetyProfile("sim", "仿真干跑：和训练一样（速度、工作空间、指尖可以碰带面）", 1.0, 0.008, None, True),
    "follow": SafetyProfile("follow", "跟随不抓：速度 ×0.3，夹爪锁定张开，TCP 不低于带面上 10 cm", 0.3, 0.10, 0.003, False),
    # 2026-11-24 抓取档指尖余量 3 → 8 mm：step5 实测关节重力方向稳态下沉 2～5 mm，3 mm 余量会让指尖蹭带面。
    #   代价（step4 --lab-belt 100 回合）：总成功率 62% → 54%，现场带速 84.6% → 76.9%。--finger-clearance-mm 可覆盖
    "grasp": SafetyProfile("grasp", "闭环抓取：速度同训练，指尖不低于带面上 8 mm，抓起后停住", 1.0, 0.008, 0.008, True),
}

JOINT_LIMIT_MARGIN_RAD = 0.03
WORKSPACE_TOL_M = 0.01
STREAM_STALE_S = 0.1


# =============================================================================
# 【说明】SafetyGuard：所有"能不能发"的判断都在这里。
#   构造时把安全几何写进孪生模型（孪生只用来算运动学，不跑物理，改它不影响任何仿真）：
#     · 带面、护栏抬到"安全带面"高度：配置 belt_surface_z（独立实测 −0.12）和标定 step5 用腕部相机
#       测得的值（−0.112）可能差 8 mm，按更高的算，宁可少下 8 mm 也不压到带
#     · 控制器的工作空间下限 = 安全带面 + 本档位 tcp_floor
#   每个 10 Hz 目标走三步（episode._servo）：
#   ① check_step(q_from, q_ik)：控制器 IK 解出的目标（修正之前，controller.last_ik_q），单拍关节变化 ≤ 限速 × 控制周期
#      （控制器已经限过，再查一次防止上游出错；修正会多动一点，那是往安全方向动，不受此限，
#        100 Hz 发送器仍按限速逐点截断）
#   ② project(q)：把目标"修正"进安全范围——修正而不是停止，策略照常跑。用的是控制器的 correct()
#      （agents/controllers/pd_ee_delta_pose.py，v6 起训练环境也在用），这里带面按"安全带面"算：
#      a. TCP 拉回工作空间：IK 要同时满足姿态，实际解可能偏出 1～3 cm（v5 干跑 100 回合有 17 回合因此被停）
#      b. 夹爪最低点低于"安全带面 + finger_clearance" → TCP 竖直抬高差值，重新解 IK；
#         夹爪落在护栏正上方的部分还要高过"护栏顶 + finger_clearance"（训练不做这一条，只罚实际接触；
#         指令目标比实际机械臂领先约 0.2 s，不修正的话 v6 张着夹爪下降时常被③当成"会碰护栏"停下）
#      c. 关节截到软限位 [下限 + 0.03, 上限 − 0.03] rad（v5 常把 J4 弯到 2.15～2.18，上限 2.18）
#   ③ check_pose(q)：修正后的目标必须满足（不满足就原地停）
#      1. 数值有限、7 维   2. 关节在软限位内   3. TCP 在工作空间里（容差 1 cm）
#      4. 机械臂/夹爪不碰带面、护栏、底座（孪生模型 mj_collision，穿透才算；finger_clearance=None 时手指碰带面不算，同训练）
#   check_path(q_from, q_to)：回合之间的关节空间直线移动（回起点），沿途 20 个点逐个 check_pose
#   check_motion(...)：每拍查运行状态——实测关节跟得上已发指令（真机执行时）；100 Hz 发送线程还在跑
#   check_camera(...)：相机有新帧
# =============================================================================
class SafetyGuard:
    def __init__(self, twin, profile: SafetyProfile, belt_z_safe: float):
        self.twin, self.profile = twin, profile
        cfg, m = twin.cfg, twin.model
        self.belt_z_safe = float(max(belt_z_safe, cfg.belt_surface_z))
        dz = self.belt_z_safe - cfg.belt_surface_z
        m.body_pos[twin.env.belt_body][2] += dz
        for g in twin.env.rail_geoms:
            m.geom_pos[g][2] += dz
        ctl = twin.controller
        ctl.ws_lo[2] = self.belt_z_safe + profile.tcp_floor_above_belt_m
        self.ws_lo, self.ws_hi = ctl.ws_lo.copy(), ctl.ws_hi.copy()
        self.q_lo, self.q_hi = ctl.ik.lo + JOINT_LIMIT_MARGIN_RAD, ctl.ik.hi - JOINT_LIMIT_MARGIN_RAD
        vmax = np.broadcast_to(np.asarray(cfg.max_joint_speed_rad_s, float), (7,))     # 孪生配置已按档位缩放
        self.max_step = vmax * twin.control_dt * 1.05
        collide = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.robot_geoms = frozenset(g for g in twin.agent.geoms if collide[g])
        self.obstacles = twin.env.obstacle_geoms
        self.finger_geoms = frozenset(twin.agent.finger_geoms)
        rails = twin.env.rail_geoms
        self.rail_inner = float(min(abs(m.geom_pos[g][0] - cfg.belt_center_x) - m.geom_size[g][0] for g in rails))
        self.rail_top_z = float(max(m.geom_pos[g][2] + m.geom_size[g][2] for g in rails))     # 已随安全带面抬高

    # ------------------------------------------------------------------ ② 修正
    def project(self, q: np.ndarray, gripper_q: float) -> tuple[np.ndarray, list[str]]:
        """Correct a target into the safe set with the controller's shared correct() (same code the training env runs),
        but with the stricter safe belt height and the rails; returns (q, what changed)."""
        c = self.profile.finger_clearance_m
        if c is None:
            return self.twin.controller.correct(q, gripper_q, None, JOINT_LIMIT_MARGIN_RAD, pullback=True)
        return self.twin.controller.correct(q, gripper_q, self.belt_z_safe + c, JOINT_LIMIT_MARGIN_RAD, pullback=True,
                                            rail=(self.twin.cfg.belt_center_x, self.rail_inner, self.rail_top_z + c))

    def gripper_lowest_z(self, q: np.ndarray, gripper_q: float) -> float:
        return self.twin.controller.gripper_lowest_z(q, gripper_q)

    # ------------------------------------------------------------------ ① ③ 检查
    def check_step(self, q_from: np.ndarray, q_to: np.ndarray) -> list[str]:
        q_to = np.asarray(q_to, float)
        if q_to.shape != (7,) or not np.all(np.isfinite(q_to)):
            return ["关节目标不是 7 个有限数"]
        step = np.abs(q_to - np.asarray(q_from, float))
        if np.any(step > self.max_step):
            j = int(np.argmax(step / self.max_step))
            return [f"单拍关节变化过大：J{j + 1} {step[j]:.3f} rad（上限 {self.max_step[j]:.3f}）"]
        return []

    def check_pose(self, q: np.ndarray, gripper_q: float) -> list[str]:
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return ["关节目标不是 7 个有限数"]
        out = []
        bad = np.where((q < self.q_lo - 1e-6) | (q > self.q_hi + 1e-6))[0]
        if len(bad):
            out.append("关节接近限位：" + "，".join(f"J{j + 1}={q[j]:+.2f}" for j in bad))
        self.twin.set_joints(q, gripper_q)
        p, _ = self.twin.tcp_pose()
        if np.any(p < self.ws_lo - WORKSPACE_TOL_M) or np.any(p > self.ws_hi + WORKSPACE_TOL_M):
            out.append(f"TCP {np.round(p, 3).tolist()} 超出工作空间（z 下限 {self.ws_lo[2]:.3f}）")
        return out + self._collisions()

    def check_target(self, q_from: np.ndarray, q_to: np.ndarray, gripper_q: float) -> list[str]:
        """check_step + check_pose on one target (scripted moves in step5, which need no projection)."""
        return self.check_step(q_from, q_to) or self.check_pose(q_to, gripper_q)

    def check_path(self, q_from: np.ndarray, q_to: np.ndarray, gripper_q: float, n: int = 20) -> list[str]:
        """A joint-space straight line (moves between episodes): check poses along it."""
        for w in np.linspace(0, 1, n + 1)[1:]:
            out = self.check_pose(q_from + w * (np.asarray(q_to) - q_from), gripper_q)
            if out:
                return [f"路径 {w * 100:.0f}% 处：" + x for x in out]
        return []

    def _collisions(self) -> list[str]:
        m, d = self.twin.model, self.twin.data
        mujoco.mj_collision(m, d)
        hits = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.robot_geoms and b in self.obstacles:
                    if self.profile.finger_clearance_m is None and a in self.finger_geoms:
                        continue                                     # 同训练：手指碰带面/护栏不算
                    hits.add(f"{m.geom(a).name} 会碰到 {m.geom(b).name}")
        return [f"目标位姿下{h}" for h in sorted(hits)]

    # ------------------------------------------------------------------ 运行状态
    def check_motion(self, now: float, q_meas: np.ndarray, streamer, executing: bool) -> list[str]:
        """Rules 6-7a: the arm follows what was sent, and the 100 Hz sender is alive."""
        out = []
        if executing and streamer.q_cmd is not None:
            err = np.abs(np.asarray(q_meas) - streamer.q_cmd)
            if err.max() > self.profile.max_tracking_err_rad:
                j = int(np.argmax(err))
                out.append(f"实测关节跟不上指令：J{j + 1} 相差 {err[j]:.2f} rad（可能被挡住或撞到东西）")
        if streamer.fault:
            out.append(streamer.fault)
        elif streamer.last_tick is not None and now - streamer.last_tick > STREAM_STALE_S:
            out.append(f"100 Hz 发送已 {now - streamer.last_tick:.2f} s 没有动作")
        return out

    def check_camera(self, now: float, frame_stamp: float | None) -> list[str]:
        """Rule 7b: frames keep coming."""
        if frame_stamp is None:
            return ["相机没有新帧"]
        if now - frame_stamp > self.profile.max_frame_age_s:
            return [f"相机 {now - frame_stamp:.2f} s 没有新帧"]
        return []
