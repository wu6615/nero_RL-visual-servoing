"""The whole dynamic-grasp task as one batched object; a simulator only has to provide states and apply targets."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...utils.config import EnvConfig
from .controller import BatchController, solve_home, solve_ik
from .kinematics import NeroKinematics, home_rotation
from .objects import PAD_GAP0_M, ObjectBatch, make_specs
from .observation import assemble_obs
from .perception import GraspMemory, WristCamera, add_noise
from .reward import dense_reward
from .so3 import quat_wxyz_to_mat, rot_z, symmetric_rot_dist
from .trajectory import BeltTrajectories

# =============================================================================
# 【说明】任务逻辑（观测、控制、奖励、终止、重置）都在这里，Isaac 环境只是"接线"：
#   每个控制步  pre_step(state, action) → 关节目标；物理步里 substep() 推进带和物体驱动；
#               post_step(state)        → 奖励、终止、这一步的观测（重置之前的，作终止观测用）
#   重置        reset(ids) → 起始关节、物体初始位姿；之后 observe(state, ids) 取新回合第一帧观测
#   所有位置都在各环境自己的机器人基座系（Isaac 环境负责减去 env_origins）。
#   和 DynamicGraspEnv 有意保留的差别（都记在 isaac/README.md）：物体规格按环境固定、遮挡用手指凸包、
#   物体与带面的摩擦取物体规格的值、手指长度随机没有做。
# =============================================================================


@dataclass
class SimState:
    q: torch.Tensor                  # (B,7)
    qd: torch.Tensor                 # (B,7)
    grip: torch.Tensor               # (B,) 单指行程（两指平均）
    obj_pos: torch.Tensor            # (B,3)
    obj_quat: torch.Tensor           # (B,4) wxyz
    obj_vel: torch.Tensor            # (B,3)
    left: torch.Tensor               # (B,) bool 左指接触物体
    right: torch.Tensor
    touch: torch.Tensor              # 机器人任何部位碰到物体（这一控制步内）
    collision: torch.Tensor          # 非手指碰带面/护栏/底座，或手指碰护栏（这一控制步内）


class GraspTask:
    def __init__(self, cfg: EnvConfig, num_envs: int, library: int, object_seed: int, seed: int = 0, device="cpu",
                 dtype=torch.float32):
        self.cfg, self.B, self.device, self.dtype = cfg, num_envs, device, dtype
        self.dt = 1.0 / cfg.control_freq
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.kin = NeroKinematics(device=device, dtype=dtype)
        self.specs = make_specs(cfg, library, object_seed)
        self.obj = ObjectBatch(self.specs, cfg, num_envs, device, dtype)
        self.q_home = solve_home(self.kin, cfg)
        self.ctl = BatchController(self.kin, cfg, num_envs, self.q_home)
        self.traj = BeltTrajectories(cfg, self.obj, num_envs, self.gen, device, dtype)
        self.cam = WristCamera(self.kin, cfg)
        self.N, self.K = cfg.grasp_obs.num_grasps, cfg.grasp_obs.num_points
        if cfg.grasp_obs.points_mode != "gauss":
            raise ValueError("Isaac 版只实现了 points_mode=gauss")
        self.mem = GraspMemory(cfg, num_envs, self.obj.G, device, dtype)
        self.prev_action = torch.zeros(num_envs, 7, device=device, dtype=dtype)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.lifted_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.attached = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.visible = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.all_ids = torch.arange(num_envs, device=device)

    # ------------------------------------------------------------ reset
    def reset(self, ids: torch.Tensor) -> torch.Tensor:
        """Start new episodes; returns start joints (len(ids), 7). Object poses come from object_drive()."""
        c, n = self.cfg, len(ids)
        self.traj.reset(ids)
        self.mem.reset(ids)
        self.prev_action[ids], self.steps[ids], self.lifted_steps[ids], self.attached[ids] = 0.0, 0, 0, True
        u = lambda *s: 2 * torch.rand(*s, generator=self.gen, device=self.device, dtype=self.dtype) - 1
        p = torch.tensor([c.belt_center_x + c.init_tcp_offset_x, c.init_tcp_offset_y, c.belt_surface_z + c.init_tcp_height],
                         device=self.device, dtype=self.dtype) + u(n, 3) * c.init_pos_noise_m
        R = rot_z(u(n) * c.init_yaw_noise_rad) @ home_rotation(self.device, self.dtype)
        q0 = solve_ik(self.kin, self.q_home.expand(n, 7), p, R, self.q_home, iters=100)
        self.ctl.reset(ids, q0)
        return q0

    # ------------------------------------------------------------ stepping
    def pre_step(self, s: SimState, action: torch.Tensor):
        """Policy action -> (previous joint command, new joint targets, finger target); interpolate between the first two."""
        a = action.to(self.dtype).clamp(-1, 1)
        q_prev = self.ctl.q_cmd.clone()
        q_tgt, grip = self.ctl.compute(s.q, s.grip, a)
        self.prev_action = a
        return q_prev, q_tgt, grip

    def substep(self, dt: float):
        self.traj.step(dt)

    def object_drive(self):
        """Where the still-attached objects must be: (pos, yaw, velocity (B,3))."""
        pos, yaw = self.traj.pose()
        vel = torch.zeros_like(pos)
        vel[:, 1] = self.traj.velocity_y()
        return pos, yaw, vel

    def note_touch(self, touch: torch.Tensor):
        self.attached &= ~touch                                  # 机器人一碰到物体就脱离驱动

    def post_step(self, s: SimState):
        """After the control step: (obs before any reset, reward, terminated, truncated, info)."""
        self.steps += 1
        obs = self.observe(s, self.all_ids)
        info = self.evaluate(s)
        reward = dense_reward(self.cfg, info, self.steps)
        term = info["fell"] | info["missed"]
        if self.cfg.terminate_on_success:
            term = term | info["success"]
        if self.cfg.terminate_on_collision:
            term = term | info["collision"]
        trunc = (self.steps >= self.cfg.max_steps) & ~term
        return obs, reward, term, trunc, info

    # ------------------------------------------------------------ observation
    def observe(self, s: SimState, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        g, o = self.cfg.grasp_obs, self.obj
        q, grip = s.q[ids], s.grip[ids]
        Rg, pg = self.kin.gripper_base(q)
        p, R = pg + Rg @ self.kin.p_tcp, Rg @ self.kin.R_tcp
        Ro, po = quat_wxyz_to_mat(s.obj_quat[ids]), s.obj_pos[ids]
        world = lambda x: x @ Ro.transpose(1, 2) + po[:, None]
        full = g.visible_rule == "full_top"                  # 规则同 DynamicGraspEnv._observe_grasps
        vis = self.cam.visible(world(o.vis_obj[ids]), Rg, pg, grip, occlusion=not full) & o.valid[ids]
        if full:
            inside = self.cam.visible(world(o.outline[ids]), Rg, pg, grip, occlusion=False).all(1, keepdim=True)
            exposed = self.cam.visible(world(o.face[ids]), Rg, pg, grip).to(self.dtype).mean(1, keepdim=True)
            vis = vis & inside & (exposed >= g.min_top_exposed)
        elif g.visible_rule != "point":
            raise ValueError(f"未知 visible_rule: {g.visible_rule}")
        seen = vis.any(1)
        T = o.T_obj[ids].clone()
        T[..., :3, :3] = Ro[:, None] @ o.T_obj[ids][..., :3, :3]
        T[..., :3, 3] = world(o.T_obj[ids][..., :3, 3])
        T = add_noise(T, g.pos_noise_m, math.radians(g.rot_noise_deg), self.gen)
        est = po + g.track_pos_noise_m * torch.randn(len(ids), 3, generator=self.gen, device=self.device, dtype=self.dtype)
        Gw, sc, use, age = self.mem.update(ids, self.steps[ids].to(self.dtype) * self.dt, T, o.scores[ids], vis, seen, est)
        self.visible[ids] = seen
        from ...grasp_detection.analytic_grasps import FINGER_HALF_OPEN
        pts = FINGER_HALF_OPEN / 3 * torch.randn(len(ids), self.K, 3, generator=self.gen, device=self.device, dtype=self.dtype)
        return assemble_obs(p, R, Gw, sc, use, seen, age, grip / self.cfg.gripper_max_q, self.prev_action[ids], pts, self.N)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, s: SimState) -> dict[str, torch.Tensor]:
        c, o = self.cfg, self.obj
        p, R, J = self.kin.tcp_jacobian(s.q)
        Ro = quat_wxyz_to_mat(s.obj_quat)
        misalign, width = o.jaw_fit((Ro.transpose(1, 2) @ R[:, :, 1:2]).squeeze(-1))
        gap = PAD_GAP0_M + 2 * s.grip
        grasped = s.left & s.right & (s.grip > 1e-3)
        if c.clean_grasp:
            grasped = grasped & (misalign <= math.radians(c.clean_align_deg)) & ((gap - width).abs() <= c.clean_width_tol_m)
        lift = s.obj_pos[:, 2] - o.half[:, 2] - c.belt_surface_z
        lifted = grasped & (lift > c.lift_height_m)
        self.lifted_steps = torch.where(lifted, self.lifted_steps + 1, 0)
        Re = R.transpose(1, 2)[:, None] @ (Ro[:, None] @ o.T_obj[..., :3, :3])
        te = o.T_obj[..., :3, 3] @ Ro.transpose(1, 2) + s.obj_pos[:, None] - p[:, None]
        dt = te.norm(dim=-1)
        dr = symmetric_rot_dist(Re) / math.pi
        cost = (dt.clamp(0.01, 0.5) + dr.clamp(0.01, 0.5)).masked_fill(~o.valid, float("inf"))
        gi = cost.argmin(1, keepdim=True)
        v_tcp = (J[:, :3] @ s.qd[..., None]).squeeze(-1)
        sign = math.copysign(1.0, c.belt_direction_sign)
        return dict(grasped=grasped, lifted=lifted, lift=lift, success=lifted & (self.lifted_steps >= 2),
                    robot_static=s.qd.abs().amax(1) < 0.2, collision=s.collision, visible=self.visible,
                    dist_t=dt.gather(1, gi).squeeze(1), dist_r=dr.gather(1, gi).squeeze(1),
                    fell=s.obj_pos[:, 2] < c.belt_surface_z - 0.05,
                    missed=~grasped & (s.obj_pos[:, 1] * sign > abs(c.downstream_limit_y)),
                    misalign=misalign, width=width, gap=gap, tilt=torch.acos((-R[:, 2, 2]).clamp(-1, 1)),
                    rel_speed=(v_tcp - s.obj_vel).norm(dim=1), closing=self.prev_action[:, 6] < 0)
