"""AgileX NERO7 + two-finger gripper bound to a loaded MuJoCo model (ids, state, contacts).

Counterpart of GAP-RL ``agents/robots/ur5e_robotiq85_old.py``.
"""
from __future__ import annotations

import mujoco
import numpy as np

# =============================================================================
# 【说明】机器人"句柄"：只负责查名字→id、读写关节、查接触，不含任务逻辑。
#   TCP = ee_site：z 轴为接近方向，y 轴为夹爪闭合方向，在指尖平面后 10 mm（robot.xml）。
#   夹爪：gripper_joint1 行程 [0, 0.05]，gripper_joint2 由等式约束对称跟随；
#         指垫中心间距 = 0.008 + 2q（q=0 闭合，q=0.05 全开）。
# =============================================================================


class Nero7Agent:
    ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
    GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
    TCP_SITE = "ee_site"
    WRIST_CAMERA = "wrist_camera"
    FINGER_GEOMS = ("left_finger_geom", "right_finger_geom")
    ROOT_BODY = "base_link"

    def __init__(self, model: mujoco.MjModel, gripper_max_q: float = 0.05):
        m = self.model = model
        self.gripper_max_q = gripper_max_q
        self.tcp_site = m.site(self.TCP_SITE).id
        self.camera = m.camera(self.WRIST_CAMERA).id
        self.arm_qadr = np.array([m.joint(j).qposadr[0] for j in self.ARM_JOINTS])
        self.arm_dadr = np.array([m.joint(j).dofadr[0] for j in self.ARM_JOINTS])
        self.grip_qadr = tuple(int(m.joint(j).qposadr[0]) for j in self.GRIPPER_JOINTS)
        self.act_arm = np.array([m.actuator(j).id for j in self.ARM_JOINTS])
        self.act_grip = m.actuator("gripper").id
        root = m.body(self.ROOT_BODY).id
        bodies = {b for b in range(m.nbody) if self._descends(b, root)}
        self.geoms = frozenset(g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies)
        self.finger_geoms = tuple(m.geom(n).id for n in self.FINGER_GEOMS)

    def _descends(self, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = self.model.body_parentid[body]
        return False

    # ---- state
    def tcp_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return data.site_xpos[self.tcp_site].copy(), data.site_xmat[self.tcp_site].reshape(3, 3).copy()

    def arm_qpos(self, data) -> np.ndarray:
        return data.qpos[self.arm_qadr].copy()

    def arm_speed(self, data) -> float:
        return float(np.max(np.abs(data.qvel[self.arm_dadr])))

    def gripper_opening(self, data) -> float:
        return float(data.qpos[self.grip_qadr[0]])

    def set_state(self, data, q_arm: np.ndarray, grip_q: float):
        data.qpos[self.arm_qadr] = q_arm
        data.ctrl[self.act_arm] = q_arm
        data.qpos[self.grip_qadr[0]], data.qpos[self.grip_qadr[1]] = grip_q, -grip_q
        data.ctrl[self.act_grip] = grip_q

    def set_ctrl(self, data, q_arm: np.ndarray, grip_q: float):
        data.ctrl[self.act_arm] = q_arm
        data.ctrl[self.act_grip] = grip_q

    # ---- contacts
    def scan_contacts(self, data, target_geom: int, obstacle_geoms: frozenset,
                      finger_obstacles=()) -> dict[str, bool]:
        """touch: any robot geom on target; left/right: finger on target;
        collision: non-finger robot geom on an obstacle (belt, rails), or a finger on one of ``finger_obstacles``."""
        touch = left = right = coll = False
        lf, rf = self.finger_geoms
        for i in range(data.ncon):
            g1, g2 = int(data.contact.geom1[i]), int(data.contact.geom2[i])
            for a, b in ((g1, g2), (g2, g1)):
                if a not in self.geoms:
                    continue
                if b == target_geom:
                    touch = True
                    left |= a == lf
                    right |= a == rf
                elif b in obstacle_geoms and (a not in (lf, rf) or b in finger_obstacles):
                    coll = True
        return dict(touch=touch, left=left, right=right, collision=coll)
