"""TCP delta-pose controller: policy action -> DLS IK -> NERO7 joint position targets.

Counterpart of GAP-RL ``agents/controllers/pd_ee_pose.py`` (pd_ee_delta_pose).
The IK runs on a private MjData so the physics state is never touched; the same
mapping is meant to be reused on hardware before the 100 Hz joint stream sender.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ...utils.config import EnvConfig
from ...utils.geometry import expmap, logmap
from ..robots.nero7 import Nero7Agent

ARM_JOINTS = Nero7Agent.ARM_JOINTS
RAIL_SIDE_TOL_M = 0.003      # 离护栏内侧面不到 3 mm 的网格点也按"在护栏上方"算


class DiffIK:
    def __init__(self, model: mujoco.MjModel, site: str = "ee_site", damping: float = 1e-3,
                 posture_gain: float = 0.05):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 0].copy()
        self.hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 1].copy()
        self.damping = damping
        self.posture_gain = posture_gain
        self.q_posture = np.clip(np.zeros(7), self.lo, self.hi)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(self, q0: np.ndarray, p_des: np.ndarray, R_des: np.ndarray, iters: int = 8,
              max_step: float | None = None, rot_weight: float = 0.5) -> tuple[np.ndarray, float, float]:
        """Return (q, pos_err_m, rot_err_rad).  ``max_step`` clips |q - q0| per joint."""
        q = np.array(q0, dtype=float)
        W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
        pos_err = rot_err = np.inf
        for _ in range(iters):
            p, R = self.fk(q)
            e = np.concatenate([p_des - p, R @ logmap(R.T @ R_des)])
            pos_err, rot_err = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if pos_err < 2e-4 and rot_err < 2e-3:
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site_id)
            J = np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])
            Jw = W @ J
            JJt = Jw @ Jw.T + self.damping * np.eye(6)
            J_pinv = Jw.T @ np.linalg.solve(JJt, np.eye(6))
            dq = J_pinv @ (W @ e)
            null = np.eye(7) - J_pinv @ Jw
            dq += null @ (self.posture_gain * (self.q_posture - q))
            q = np.clip(q + dq, self.lo, self.hi)
            if max_step is not None:
                q = np.clip(q, q0 - max_step, q0 + max_step)
        p, R = self.fk(q)
        pos_err = float(np.linalg.norm(p_des - p))
        rot_err = float(np.linalg.norm(logmap(R.T @ R_des)))
        return q, pos_err, rot_err


# =============================================================================
# 【说明】动作定义（7 维，均归一化到 [-1, 1]）：
#   a[0:3] 平移增量 × max_delta_pos_m；a[3:6] 旋转向量增量 × max_delta_rot_rad
#          action_frame="ee" 时在 TCP 系下表达（眼在手上更自然），"base" 时在基座系
#   a[6]   夹爪目标开口：-1 闭合，+1 全开（绝对量，不是增量）
# 指令累积方式（与真机伺服流一致）：目标位姿 = 上一步"指令"位姿 ⊕ 增量，而不是实测位姿 ⊕ 增量。
#   位置环跟随有滞后，若每步从实测位姿出发，稳态速度只有指令的一半左右。
#   leash：指令与实测相差超过 cmd_leash_m / cmd_leash_rad（例如被物体挡住）时，把指令拉回实测附近，防止跑飞；
#          默认取 max(3 cm, 1.5 × 最大步长)，旋转同理。
# 之后裁剪到工作空间，用 DiffIK（以上一步关节指令为初值）求关节目标，单步变化 ≤ max_joint_speed × 控制周期，
# 最后经 correct() 修正（配置里打开时）。
# =============================================================================
class EEDeltaPoseController:
    def __init__(self, agent: Nero7Agent, cfg: EnvConfig, control_dt: float,
                 cmd_leash_m: float | None = None, cmd_leash_rad: float | None = None):
        self.agent = agent
        self.cfg = cfg
        self.control_dt = control_dt
        self.ik = DiffIK(agent.model, site=agent.TCP_SITE)
        self.ws_lo = np.full(3, -np.inf)
        self.ws_hi = np.full(3, np.inf)
        # leash 至少 1.5 个最大步长：步长调大（如 5 cm/步）后，位置环滞后会超过固定 3 cm，牵引绳会把速度卡住
        self.leash_m = cmd_leash_m if cmd_leash_m is not None else max(0.03, 1.5 * cfg.max_delta_pos_m)
        self.leash_rad = cmd_leash_rad if cmd_leash_rad is not None else max(0.15, 1.5 * cfg.max_delta_rot_rad)
        self.p_cmd = self.R_cmd = self.q_cmd = None
        self.last_fixes: list[str] = []
        self.last_ik_q: np.ndarray | None = None
        self._gripper_verts = self._gripper_vertices()

    def set_workspace(self, lo, hi):
        self.ws_lo, self.ws_hi = np.asarray(lo, float), np.asarray(hi, float)

    def reset(self, data):
        """Align the command with the measured state (call at episode start)."""
        self.p_cmd, self.R_cmd = self.agent.tcp_pose(data)
        self.q_cmd = self.agent.arm_qpos(data)

    def target_pose(self, p: np.ndarray, R: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        dpos, drot = action[:3] * c.max_delta_pos_m, action[3:6] * c.max_delta_rot_rad
        if c.action_frame == "ee":
            p_des, R_des = p + R @ dpos, R @ expmap(drot)
        elif c.action_frame == "base":
            p_des, R_des = p + dpos, expmap(drot) @ R
        else:
            raise ValueError(f"未知 action_frame: {c.action_frame}")
        return np.clip(p_des, self.ws_lo, self.ws_hi), R_des

    def compute(self, data, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (arm joint targets, gripper joint target) for one control step."""
        if self.p_cmd is None:
            self.reset(data)
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        p_meas, R_meas = self.agent.tcp_pose(data)
        gap = self.p_cmd - p_meas
        if np.linalg.norm(gap) > self.leash_m:
            self.p_cmd = p_meas + gap * self.leash_m / np.linalg.norm(gap)
        if np.linalg.norm(logmap(R_meas.T @ self.R_cmd)) > self.leash_rad:
            self.R_cmd = R_meas
        p_des, R_des = self.target_pose(self.p_cmd, self.R_cmd, a)
        q_tgt, _, _ = self.ik.solve(self.q_cmd, p_des, R_des, iters=6,
                                    max_step=np.asarray(self.cfg.max_joint_speed_rad_s, float) * self.control_dt)
        grip = (a[6] + 1) / 2 * self.agent.gripper_max_q
        self.last_ik_q = q_tgt.copy()                       # 修正前的 IK 结果（部署时安全层用它查单拍步长）
        c = self.cfg
        floor = None if c.finger_clearance_m is None else c.belt_surface_z + c.finger_clearance_m
        q_tgt, self.last_fixes = self.correct(q_tgt, max(grip, self.agent.gripper_opening(data)), floor,
                                              c.joint_limit_margin_rad, c.workspace_pullback)
        self.q_cmd = q_tgt
        self.p_cmd, self.R_cmd = self.ik.fk(q_tgt)          # 实际可达的指令位姿（受限位/步长/修正影响）
        return q_tgt, float(grip)

    # =========================================================================
    # 【说明】目标修正 correct()：IK 解出的关节目标，发出（仿真里写进执行器 / 真机上进 100 Hz 发送）之前依次
    #   a. 拉回工作空间：期望位置已经截在工作空间里，但 IK 要同时满足姿态，实际解可能偏出 1～3 cm
    #      → 朝截好的位置再解一次 IK
    #   b. 夹爪离带面：夹爪（指尖、夹爪座）网格最低点低于 finger_floor_z → TCP 竖直抬高差值，再解 IK（最多 3 次）。
    #      不能只限 TCP 高度：TCP 在指尖平面后 10 mm，但张开的夹爪倾斜 15° 时外侧指尖比 TCP 低 24 mm
    #   c. 关节截到软限位 [下限 + margin, 上限 − margin]
    #   训练（v6 起）：环境用配置里的带面高度 + finger_clearance_m，策略学的就是"带修正的手"——
    #     v5 训练时没有修正，学会了把指尖顶在带面上（仿真里带面把手指挡住、不扣分），部署时被修正打断。
    #   部署：deploy/runtime/safety.py 用同一个函数，带面按更保守的"安全带面"算，并且给 rail：
    #     夹爪上落在护栏正上方（离带中线 ≥ 护栏内侧面 − 3 mm）的网格点还要高过护栏顶 + 间隙，不够也竖直抬高。
    #     训练不加这一条（rail=None）：训练只罚实际碰到护栏，v6 在训练环境里 100 回合实际碰护栏 0 次；
    #     但指令目标比实际机械臂领先约 0.2 s，张着夹爪下降时外侧指尖的目标会伸进护栏。部署按指令目标查碰撞，
    #     不修正的话这些目标会被当成"会碰护栏"直接停（v6 闭环干跑 100 回合停了 20 次）。
    #   返回 (修正后的 q, 修正说明列表)；夹爪开口按张得更开的算（倾斜时指尖更低）。
    # =========================================================================
    def correct(self, q: np.ndarray, gripper_q: float, finger_floor_z: float | None, joint_margin_rad: float = 0.0,
                pullback: bool = False, rail: tuple[float, float, float] | None = None) -> tuple[np.ndarray, list[str]]:
        """``rail`` = (带中线 x, 护栏内侧面离中线的距离, 护栏上方网格点要达到的高度)；None = 不管护栏（训练）."""
        fixes, ik = [], self.ik
        if pullback:
            p, R = ik.fk(q)
            p_in = np.clip(p, self.ws_lo, self.ws_hi)
            off = float(np.linalg.norm(p - p_in))
            if off > 1e-3:
                q, _, _ = ik.solve(q, p_in, R, iters=10)
                fixes.append(f"TCP 拉回工作空间 {off * 1000:.0f} mm")
        if finger_floor_z is not None:
            raised, over_rail = 0.0, False
            for _ in range(3):
                deficit, by_rail = self.finger_deficit(q, gripper_q, finger_floor_z, rail)
                if deficit <= 5e-4:
                    break
                over_rail |= by_rail
                p, R = ik.fk(q)
                q, _, _ = ik.solve(q, p + [0.0, 0.0, deficit + 1e-3], R, iters=10)
                raised += deficit + 1e-3
            if raised > 0:
                fixes.append(f"指尖离{'护栏' if over_rail else '带面'}不够，目标抬高 {raised * 1000:.0f} mm")
        if joint_margin_rad > 0:
            clipped = np.clip(q, ik.lo + joint_margin_rad, ik.hi - joint_margin_rad)
            bad = np.where(np.abs(clipped - q) > 1e-9)[0]
            if len(bad):
                fixes.append("截到软限位：" + "，".join(f"J{j + 1}" for j in bad))
            q = clipped
        return q, fixes

    def gripper_points(self, q: np.ndarray, gripper_q: float) -> np.ndarray:
        """World positions of all gripper mesh vertices (fingers + gripper body) at joints ``q`` (IK's private MjData)."""
        d, (g1, g2) = self.ik.data, self.agent.grip_qadr
        d.qpos[g1], d.qpos[g2] = gripper_q, -gripper_q
        self.ik.fk(q)
        return np.concatenate([V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, V in self._gripper_verts])

    def gripper_lowest_z(self, q: np.ndarray, gripper_q: float) -> float:
        return float(self.gripper_points(q, gripper_q)[:, 2].min())

    def finger_deficit(self, q: np.ndarray, gripper_q: float, floor_z: float,
                       rail: tuple[float, float, float] | None = None) -> tuple[float, bool]:
        """How far the gripper must rise to clear ``floor_z`` everywhere and the rail top above the rails;
        returns (deficit, whether a rail set it)."""
        P = self.gripper_points(q, gripper_q)
        belt = float(floor_z - P[:, 2].min())
        if rail is None:
            return belt, False
        cx, inner, top_z = rail
        over = np.abs(P[:, 0] - cx) >= inner - RAIL_SIDE_TOL_M
        side = float(top_z - P[over, 2].min()) if over.any() else -np.inf
        return max(belt, side), side > belt

    def _gripper_vertices(self) -> list[tuple[int, np.ndarray]]:
        """Local mesh vertices of every colliding mesh geom under gripper_base."""
        m = self.agent.model
        gb = m.body("gripper_base").id
        out = []
        for g in self.agent.geoms:
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            b = m.geom_bodyid[g]
            while b > 0 and b != gb:
                b = m.body_parentid[b]
            if b == gb:
                mid = m.geom_dataid[g]
                out.append((g, m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()))
        return out
