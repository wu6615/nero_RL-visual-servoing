"""Batched TCP delta-pose controller: port of agents/controllers/pd_ee_delta_pose.py (same maths, same constants)."""
from __future__ import annotations

import torch

from ...utils.config import EnvConfig
from .kinematics import NeroKinematics
from .so3 import expmap, logmap

# =============================================================================
# 【说明】部署用的是 MuJoCo 那一份控制器，这里必须和它算得一样，否则就多了一层仿真与真机的差别。
#   验收：tests/test_isaac_core.py 随机状态和动作，关节目标差 < 1e-3 rad。
#   和 numpy 版的唯一写法区别：逐环境的"收敛就停"用掩码实现（收敛的环境不再更新）。
# =============================================================================


def solve_ik(kin: NeroKinematics, q0, p_des, R_des, q_posture, iters: int, max_step=None, rot_weight: float = 0.5,
             damping: float = 1e-3, posture_gain: float = 0.05):
    """Damped least squares with a posture null space. Returns q (B,7)."""
    q = q0.clone()
    active = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
    W = torch.tensor([1, 1, 1, rot_weight, rot_weight, rot_weight], device=q.device, dtype=q.dtype)
    eye6 = torch.eye(6, device=q.device, dtype=q.dtype)
    eye7 = torch.eye(7, device=q.device, dtype=q.dtype)
    for _ in range(iters):
        p, R, J = kin.tcp_jacobian(q)
        e = torch.cat([p_des - p, (R @ logmap(R.transpose(1, 2) @ R_des)[..., None]).squeeze(-1)], -1)
        active = active & ~((e[:, :3].norm(dim=1) < 2e-4) & (e[:, 3:].norm(dim=1) < 2e-3))
        Jw = W[None, :, None] * J
        JJt = Jw @ Jw.transpose(1, 2) + damping * eye6
        pinv = Jw.transpose(1, 2) @ torch.linalg.inv(JJt)
        dq = (pinv @ (W * e)[..., None]).squeeze(-1)
        null = eye7 - pinv @ Jw
        dq = dq + (null @ (posture_gain * (q_posture - q))[..., None]).squeeze(-1)
        qn = torch.minimum(torch.maximum(q + dq, kin.lo), kin.hi)
        if max_step is not None:
            qn = torch.minimum(torch.maximum(qn, q0 - max_step), q0 + max_step)
        q = torch.where(active[:, None], qn, q)
    return q


class BatchController:
    def __init__(self, kin: NeroKinematics, cfg: EnvConfig, num_envs: int, q_posture: torch.Tensor):
        self.kin, self.cfg, self.q_posture = kin, cfg, q_posture
        self.dt = 1.0 / cfg.control_freq
        self.leash_m = max(0.03, 1.5 * cfg.max_delta_pos_m)
        self.leash_rad = max(0.15, 1.5 * cfg.max_delta_rot_rad)
        t = lambda x: torch.as_tensor(x, device=kin.device, dtype=kin.dtype)
        z0 = cfg.belt_surface_z
        self.ws_lo = t([cfg.workspace_x[0], cfg.workspace_y[0], z0 + cfg.workspace_z_above_belt[0]])
        self.ws_hi = t([cfg.workspace_x[1], cfg.workspace_y[1], z0 + cfg.workspace_z_above_belt[1]])
        self.max_step = t(cfg.max_joint_speed_rad_s).expand(7) * self.dt
        self.p_cmd = torch.zeros(num_envs, 3, device=kin.device, dtype=kin.dtype)
        self.R_cmd = torch.eye(3, device=kin.device, dtype=kin.dtype).repeat(num_envs, 1, 1)
        self.q_cmd = torch.zeros(num_envs, 7, device=kin.device, dtype=kin.dtype)

    def reset(self, ids: torch.Tensor, q: torch.Tensor):
        self.q_cmd[ids] = q
        self.p_cmd[ids], self.R_cmd[ids] = self.kin.tcp(q)

    def compute(self, q_meas, grip_meas, action):
        """q_meas (B,7), grip_meas (B,) per-finger travel, action (B,7) in [-1,1] -> (joint targets, finger target)."""
        c, kin = self.cfg, self.kin
        a = action.clamp(-1, 1)
        p_m, R_m = kin.tcp(q_meas)
        gap = self.p_cmd - p_m
        d = gap.norm(dim=1, keepdim=True)
        p_c = torch.where(d > self.leash_m, p_m + gap * self.leash_m / d.clamp_min(1e-9), self.p_cmd)
        ang = logmap(R_m.transpose(1, 2) @ self.R_cmd).norm(dim=1)
        R_c = torch.where((ang > self.leash_rad)[:, None, None], R_m, self.R_cmd)
        dpos, drot = a[:, :3] * c.max_delta_pos_m, expmap(a[:, 3:6] * c.max_delta_rot_rad)
        if c.action_frame == "ee":
            p_des, R_des = p_c + (R_c @ dpos[..., None]).squeeze(-1), R_c @ drot
        elif c.action_frame == "base":
            p_des, R_des = p_c + dpos, drot @ R_c
        else:
            raise ValueError(f"未知 action_frame: {c.action_frame}")
        p_des = torch.minimum(torch.maximum(p_des, self.ws_lo), self.ws_hi)
        q = solve_ik(kin, self.q_cmd, p_des, R_des, self.q_posture, iters=6, max_step=self.max_step)
        grip = (a[:, 6] + 1) / 2 * c.gripper_max_q
        q = self.correct(q, torch.maximum(grip, grip_meas))
        self.q_cmd = q
        self.p_cmd, self.R_cmd = kin.tcp(q)
        return q, grip

    def correct(self, q, grip_q):
        """Workspace pullback → keep the gripper off the belt → soft joint limits (see the numpy correct())."""
        c, kin = self.cfg, self.kin
        if c.workspace_pullback:
            p, R = kin.tcp(q)
            p_in = torch.minimum(torch.maximum(p, self.ws_lo), self.ws_hi)
            off = (p - p_in).norm(dim=1) > 1e-3
            if off.any():
                q = torch.where(off[:, None], solve_ik(kin, q, p_in, R, self.q_posture, iters=10), q)
        if c.finger_clearance_m is not None:
            floor = c.belt_surface_z + c.finger_clearance_m
            for _ in range(3):
                deficit = floor - kin.gripper_lowest_z(q, grip_q)
                low = deficit > 5e-4
                if not low.any():
                    break
                p, R = kin.tcp(q)
                up = torch.zeros_like(p)
                up[:, 2] = deficit + 1e-3
                q = torch.where(low[:, None], solve_ik(kin, q, p + up, R, self.q_posture, iters=10), q)
        if c.joint_limit_margin_rad > 0:
            m = c.joint_limit_margin_rad
            q = torch.minimum(torch.maximum(q, kin.lo + m), kin.hi - m)
        return q


def solve_home(kin: NeroKinematics, cfg: EnvConfig):
    """Home joints: same seeds and the same choice rule as BaseEnv._solve_home. Returns (7,)."""
    import numpy as np
    lo, hi = kin.lo.cpu().numpy().astype(float), kin.hi.cpu().numpy().astype(float)
    rng = np.random.default_rng(7)
    first = np.clip(np.asarray(cfg.home_seed_joint_rad, float), lo, hi)
    seeds = [first] + [first + rng.normal(0, 0.3, 7) for _ in range(23)] + [lo + (hi - lo) * rng.uniform(0.2, 0.8, 7) for _ in range(24)]
    s = torch.as_tensor(np.clip(np.stack(seeds), lo, hi), device=kin.device, dtype=kin.dtype)
    p = torch.tensor([cfg.belt_center_x + cfg.init_tcp_offset_x, cfg.init_tcp_offset_y,
                      cfg.belt_surface_z + cfg.init_tcp_height], device=kin.device, dtype=kin.dtype).expand(len(s), 3)
    from .kinematics import home_rotation
    R = home_rotation(kin.device, kin.dtype).expand(len(s), 3, 3)
    q = solve_ik(kin, s, p, R, s, iters=200)
    pf, Rf = kin.tcp(q)
    ok = ((pf - p).norm(dim=1) <= 2e-3) & (logmap(Rf.transpose(1, 2) @ R).norm(dim=1) <= 2e-2)
    if not ok.any():
        raise RuntimeError("初始位姿 IK 未收敛，检查 belt_* / init_tcp_* 配置")
    margin = torch.minimum(q - kin.lo, kin.hi - q) / (kin.hi - kin.lo)
    cost = ((0.5 - margin) ** 2).sum(1).masked_fill(~ok, float("inf"))
    return q[cost.argmin()]
