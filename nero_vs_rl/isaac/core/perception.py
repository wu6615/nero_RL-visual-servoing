"""Batched simulated perception: pinhole visibility, noise, grasp memory and velocity extrapolation."""
from __future__ import annotations

import numpy as np
import torch

from ...utils.config import EnvConfig
from .kinematics import NeroKinematics
from .so3 import expmap, quat_wxyz_to_mat

# =============================================================================
# 【说明】不渲染相机（几百个环境渲染深度图会占满显存），按几何判断"这一帧看不看得到"：
#   投影在画面内 + 深度在量程内 + 没被手指挡住。前两条和 sensors/camera.py 完全一样；
#   遮挡 MuJoCo 版用射线打碰撞网格，这里用两指的凸包做线段求交（凸包比网格略大，偏保守）；只算手指，不算夹爪座和手臂。
# =============================================================================


class WristCamera:
    def __init__(self, kin: NeroKinematics, cfg: EnvConfig):
        c = cfg.camera
        if c.pos_in_gripper_base is None or c.quat_in_gripper_base is None:
            raise ValueError("Isaac 版需要标定后的相机位姿（dc1_site.yaml）")
        t = lambda x: torch.as_tensor(np.asarray(x, float), device=kin.device, dtype=kin.dtype)
        self.kin, self.c, self.g_max = kin, c, cfg.gripper_max_q
        self.p, self.R = t(c.pos_in_gripper_base), quat_wxyz_to_mat(t(c.quat_in_gripper_base))
        if c.intrinsics is not None:
            self.fx, self.fy, self.cx, self.cy = map(float, c.intrinsics)
        else:
            self.fx = self.fy = (c.height / 2) / np.tan(np.deg2rad(c.fovy_deg) / 2)
            self.cx, self.cy = c.width / 2, c.height / 2

    def pose(self, Rg, pg):
        return pg + Rg @ self.p, Rg @ self.R

    def visible(self, pts_w: torch.Tensor, Rg, pg, grip_q: torch.Tensor, occlusion: bool = True) -> torch.Tensor:
        """pts_w (B,M,3) -> (B,M) bool; Rg/pg = gripper_base pose."""
        c = self.c
        cp, cR = self.pose(Rg, pg)
        pc = (pts_w - cp[:, None]) @ cR
        depth = -pc[..., 2]
        safe = depth.clamp_min(1e-6)
        u, v = self.cx + self.fx * pc[..., 0] / safe, self.cy - self.fy * pc[..., 1] / safe
        m = c.image_margin_px
        ok = ((depth >= c.min_depth_m) & (depth <= c.max_depth_m) & (u >= m) & (u <= c.width - m) & (v >= m) & (v <= c.height - m))
        if occlusion and c.occlusion_check:
            ok = ok & ~self.finger_blocks(pts_w, Rg, pg, grip_q)
        return ok

    def finger_blocks(self, pts_w, Rg, pg, grip_q):
        pl = (pts_w - pg[:, None]) @ Rg                                  # gripper_base 系
        o, d = self.p, pl - self.p
        hit = torch.zeros(pts_w.shape[:2], dtype=torch.bool, device=pts_w.device)
        for planes, sl in ((self.kin.planes_f1, self.kin.slide1), (self.kin.planes_f2, self.kin.slide2)):
            n, c = planes[:, :3], planes[:, 3]
            # 手指平移 g·sl 等价于起点反向平移；线段 o + s·d，s∈[0,1]，与各半空间求交
            h0 = (o - grip_q[:, None] * sl) @ n.T + c                      # (B,F) 起点到各面的有符号距离（> 0 在外侧）
            nd = d @ n.T                                                   # (B,M,F)
            ratio = -h0[:, None] / torch.where(nd.abs() < 1e-12, torch.full_like(nd, 1e-12), nd)
            inf = torch.full_like(ratio, float("inf"))
            enter = torch.where(nd < 0, ratio, -inf).amax(-1)
            leave = torch.where(nd > 0, ratio, inf).amin(-1)
            outside = ((nd.abs() < 1e-12) & (h0[:, None] > 0)).any(-1)      # 平行且在外侧：不可能相交
            hit |= (enter < leave) & (leave > 0) & (enter < 1 - 0.003 / d.norm(dim=-1).clamp_min(1e-6)) & ~outside
        return hit


class GraspMemory:
    """Batched GraspTracker (envs/observation.py): memory, least-squares velocity, extrapolation."""

    def __init__(self, cfg: EnvConfig, num_envs: int, G: int, device, dtype):
        self.g, self.dt = cfg.grasp_obs, 1.0 / cfg.control_freq
        W = self.g.vel_window
        z = lambda *s: torch.zeros(num_envs, *s, device=device, dtype=dtype)
        self.T, self.sc = z(G, 4, 4), z(G)
        self.ok = torch.zeros(num_envs, G, dtype=torch.bool, device=device)
        self.have = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.age = torch.full((num_envs,), self.g.memory_max_age_steps + 1, dtype=torch.long, device=device)
        self.ts, self.ps, self.n = z(W), z(W, 3), torch.zeros(num_envs, dtype=torch.long, device=device)
        self.vel = z(3)

    def reset(self, ids):
        self.have[ids], self.ok[ids] = False, False
        self.age[ids], self.n[ids] = self.g.memory_max_age_steps + 1, 0
        self.vel[ids] = 0.0

    def update(self, ids, t, Gw, sc, ok, seen, obj_pos):
        """Rows are the envs in ids. seen (b,) bool: anything visible this frame."""
        s = ids[seen]
        if len(s):
            self.T[s], self.sc[s], self.ok[s] = Gw[seen], sc[seen], ok[seen]
            self.have[s], self.age[s] = True, 0
            self.ts[s] = torch.cat([self.ts[s, 1:], t[seen, None]], 1)            # 窗口右端是最新
            self.ps[s] = torch.cat([self.ps[s, 1:], obj_pos[seen, None]], 1)
            self.n[s] = (self.n[s] + 1).clamp_max(self.g.vel_window)
            W = self.g.vel_window
            w = (torch.arange(W, device=s.device)[None] >= W - self.n[s, None]).to(self.ts.dtype)
            cnt = w.sum(1, keepdim=True)
            tc = (self.ts[s] - (self.ts[s] * w).sum(1, keepdim=True) / cnt) * w
            pm = (self.ps[s] * w[..., None]).sum(1, keepdim=True) / cnt[..., None]
            v = (tc[..., None] * (self.ps[s] - pm)).sum(1) / (tc ** 2).sum(1, keepdim=True).clamp_min(1e-9)
            self.vel[s] = torch.where((self.n[s] >= 3)[:, None], v, self.vel[s])
        self.age[ids[~seen]] += 1
        g = self.g
        T, use = self.T[ids].clone(), self.ok[ids].clone()
        drop = ~seen & ((g.memory_mode == "none") | ~self.have[ids] | (self.age[ids] > g.memory_max_age_steps))
        use[drop] = False
        if g.memory_mode == "extrapolate":
            T[..., :3, 3] += (self.vel[ids] * self.dt * self.age[ids, None].to(T.dtype))[:, None]      # 看得到时 age = 0
        age = (self.age[ids].to(T.dtype) / max(g.memory_max_age_steps, 1)).clamp_max(1.0)
        return T, self.sc[ids], use, age


def add_noise(T, pos_std, rot_std_rad, gen):
    n = lambda *s: torch.randn(*s, generator=gen, device=T.device, dtype=T.dtype)
    out = T.clone()
    out[..., :3, 3] += pos_std * n(*T.shape[:2], 3)
    out[..., :3, :3] = expmap(rot_std_rad * n(*T.shape[:2], 3)) @ T[..., :3, :3]
    return out
