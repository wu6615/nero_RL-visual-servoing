"""Grasp memory / velocity extrapolation and observation assembly, shared by the sim env and the real robot.

The simulation (``envs/dynamic_grasp.py``) feeds it noisy ground-truth grasps; the real robot
(``sim2real/deploy/``) feeds it grasps detected in DC1 depth.  Everything after "which grasps
did the camera see this frame" is this one code path, so real observations are built exactly
like the ones the policy was trained on.
"""
from __future__ import annotations

import numpy as np

from ..grasp_detection import analytic_grasps as gr
from ..utils.config import GraspObsConfig
from ..utils.geometry import inv_T, make_T, rot6d, symmetric_grasp_rot_dist


# =============================================================================
# 【说明】抓取记忆与速度外推（GAP-RL Graspable Region Explorer 的时间跟踪部分）
#   update(t, Gw, scores, obj_pos)：这一帧看到了抓取 → 记下来，并把物体位置加入测速窗口
#     测速：最近 vel_window 帧物体位置对时间做最小二乘直线拟合，斜率 = 速度
#     （逐帧差分会把 mm 级噪声放大成 cm/s）
#   update(t, None, ...)：这一帧没看到 → 记忆年龄 +1；
#     memory_mode=extrapolate 时用"上次看到的抓取 + 估计速度 × 控制周期 × 年龄"外推，
#     超过 memory_max_age_steps 后清空
#   年龄按控制步计（10 Hz），真机控制循环也必须按固定 10 Hz 调用。
# =============================================================================
class GraspTracker:
    def __init__(self, cfg: GraspObsConfig, control_dt: float):
        self.cfg, self.control_dt = cfg, control_dt
        self.reset()

    def reset(self):
        self.mem_grasps_world, self.mem_scores = None, None
        self.mem_age = self.cfg.memory_max_age_steps + 1
        self.est_obj_pos, self.est_obj_vel, self.track = None, np.zeros(3), []

    def update(self, t: float, grasps_world: np.ndarray | None, scores: np.ndarray | None = None,
               obj_pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (grasps_world (M,4,4), scores (M,), visible_this_frame)."""
        g = self.cfg
        if grasps_world is not None and len(grasps_world):
            if obj_pos is not None:
                self.track.append((t, np.asarray(obj_pos, float)))
                self.track = self.track[-g.vel_window:]
                if len(self.track) >= 3:
                    ts = np.array([s[0] for s in self.track])
                    ps = np.array([s[1] for s in self.track])
                    tc = ts - ts.mean()
                    self.est_obj_vel = (tc[:, None] * (ps - ps.mean(0))).sum(0) / max((tc ** 2).sum(), 1e-9)
                self.est_obj_pos = obj_pos
            self.mem_grasps_world, self.mem_scores, self.mem_age = grasps_world, scores, 0
            return grasps_world, scores, True
        self.mem_age += 1
        if g.memory_mode == "none" or self.mem_grasps_world is None or self.mem_age > g.memory_max_age_steps:
            return np.zeros((0, 4, 4)), np.zeros(0), False
        Gw = self.mem_grasps_world.copy()
        if g.memory_mode == "extrapolate":
            Gw[:, :3, 3] += self.est_obj_vel * self.control_dt * self.mem_age
        return Gw, self.mem_scores, False

    @property
    def age_norm(self) -> float:
        return min(self.mem_age / max(self.cfg.memory_max_age_steps, 1), 1.0)


# =============================================================================
# 【说明】观测字典（与 GAP-RL state_egopoints 对应，字段含义见 DynamicGraspEnv.__init__）：
#   抓取从基座系变到 TCP 系 → 按"平移 + 0.1 × 对称旋转距离"取最近的 N 个（GAP-RL near 选取）
#   → 把 K 个参考夹爪点摆到每个抓取位姿上得到 gripper_pts_diff，空槽位全部置零。
#   einsum("kj,nij->nki")：对第 n 个抓取把每个点 p ↦ R_n·p，再加平移 T_n。
# =============================================================================
def assemble_obs(tcp_p: np.ndarray, tcp_R: np.ndarray, grasps_world: np.ndarray, scores: np.ndarray,
                 visible: bool, age_norm: float, gripper_pos: float, prev_action: np.ndarray,
                 pts: np.ndarray, num_grasps: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return (obs dict, grasps_ee (N,4,4), valid (N,))."""
    N = num_grasps
    T_ew = inv_T(make_T(tcp_R, tcp_p))
    grasps_ee, sc, valid = np.zeros((N, 4, 4)), np.zeros(N), np.zeros(N, dtype=bool)
    if len(grasps_world):
        Ge = np.einsum("ij,njk->nik", T_ew, grasps_world)
        dist = np.linalg.norm(Ge[:, :3, 3], axis=1) + 0.1 * symmetric_grasp_rot_dist(Ge[:, :3, :3])
        order = np.argsort(dist)[:N]
        grasps_ee[: len(order)], sc[: len(order)], valid[: len(order)] = Ge[order], scores[order], True
    diff = np.einsum("kj,nij->nki", pts, grasps_ee[:, :3, :3]) + grasps_ee[:, None, :3, 3]
    diff[~valid] = 0.0
    obs = {
        "tcp_pose": np.concatenate([tcp_p, rot6d(tcp_R)]).astype(np.float32),
        "gripper_pos": np.array([gripper_pos], dtype=np.float32).clip(0, 1),
        "action": np.asarray(prev_action, dtype=np.float32),
        "grasp_vis": np.array([float(visible), age_norm], dtype=np.float32),
        "origin_gripper_pts": pts.astype(np.float32),
        "gripper_pts_diff": diff.astype(np.float32),
        "grasp_scores": sc.astype(np.float32),
    }
    return obs, grasps_ee, valid


def gripper_points(cfg: GraspObsConfig, k: int, rng: np.random.Generator, fixed: np.ndarray) -> np.ndarray:
    return gr.sample_gripper_points(cfg.points_mode, k, rng, fixed)
