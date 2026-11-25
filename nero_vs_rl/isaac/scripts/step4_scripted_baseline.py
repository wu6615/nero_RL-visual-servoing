"""Step 4 (server) - privileged scripted grasp: the task must be solvable before any training.

  python -m nero_vs_rl.isaac.scripts.step4_scripted_baseline --headless --num-envs 128
"""
from __future__ import annotations

import argparse
import sys

from ._app import finish, start


def scripted_action(task, s, phase, hold):
    """Above the best true grasp -> descend -> close -> lift. Returns (action, phase, hold)."""
    import torch
    from ..core.so3 import logmap, quat_wxyz_to_mat, symmetric_rot_dist
    c, o = task.cfg, task.obj
    p, R = task.kin.tcp(s.q)
    Ro = quat_wxyz_to_mat(s.obj_quat)
    Rw = Ro[:, None] @ o.T_obj[..., :3, :3]
    tw = o.T_obj[..., :3, 3] @ Ro.transpose(1, 2) + s.obj_pos[:, None]
    Re = R.transpose(1, 2)[:, None] @ Rw
    cost = (symmetric_rot_dist(Re) + 4 * (1 - o.scores) + 2 * o.T_obj[..., :2, 3].norm(dim=-1) * 10).masked_fill(~o.valid, float("inf"))
    i = cost.argmin(1)
    b = task.all_ids
    tgt, Rg = tw[b, i].clone(), Rw[b, i]
    tgt = tgt + s.obj_vel * 0.25                                                     # 补偿跟随滞后
    up = torch.where(phase == 0, 0.08, 0.0).to(p.dtype)
    tgt[:, 2] += up
    floor = c.belt_surface_z + (c.finger_clearance_m or 0.0) + 0.012
    tgt[:, 2] = tgt[:, 2].clamp_min(floor)
    lift = phase == 3
    tgt[lift] = p[lift] + torch.tensor([0.0, 0.0, 0.03], device=p.device)
    flip = torch.tensor([-1.0, -1.0, 1.0], device=p.device)
    e1, e2 = logmap(R.transpose(1, 2) @ Rg), logmap(R.transpose(1, 2) @ (Rg * flip))
    rot = torch.where((e1.norm(dim=1) <= e2.norm(dim=1))[:, None], e1, e2)
    rot[lift] = 0.0
    d = ((tgt - p)[:, None] @ R).squeeze(1)
    a = torch.zeros(task.B, 7, device=p.device)
    a[:, :3] = (0.8 * d / c.max_delta_pos_m).clamp(-1, 1)
    a[:, 3:6] = (0.8 * rot / c.max_delta_rot_rad).clamp(-1, 1)
    a[:, 6] = torch.where(phase >= 2, -1.0, 1.0)
    err, ang = d.norm(dim=1), rot.norm(dim=1)
    hold = torch.where(phase == 2, hold + 1, 0)
    phase = torch.where((phase == 0) & (err < 0.02) & (ang < 0.15), 1, phase)
    phase = torch.where((phase == 1) & (err < 0.012), 2, phase)
    phase = torch.where((phase == 2) & (hold >= 5), 3, phase)
    return a, phase, hold


def main():
    ap = argparse.ArgumentParser(description="特权脚本基线")
    ap.add_argument("--episodes", type=int, default=256)
    args, app, tc, ic = start(ap)
    code = 1
    try:
        import torch
        from ...utils.config import SITE_BELT_SPEED_MAX
        from ..sim.grasp_env import make_env
        env = make_env(tc, ic, args.num_envs or 128, args.device, args.seed)
        env.reset()
        z = lambda: torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        phase, hold, ever, n, ok = z(), z(), z().bool(), 0, []
        speed = env.task.traj.v0.clone()
        while n < args.episodes and app.is_running():
            a, phase, hold = scripted_action(env.task, env._state(), phase, hold)
            _, _, term, trunc, _ = env.step(a)
            ever |= env.task_info["success"]
            done = term | trunc
            for i in done.nonzero().squeeze(-1).tolist():
                ok.append((bool(ever[i]), float(speed[i])))
                n += 1
            phase[done], hold[done], ever[done] = 0, 0, False
            speed[done] = env.task.traj.v0[done]
        rate = sum(x for x, _ in ok) / len(ok)
        site = [x for x, v in ok if v <= SITE_BELT_SPEED_MAX]
        print(f"脚本基线 {len(ok)} 回合：成功 {rate:.1%}；现场带速 ≤ {SITE_BELT_SPEED_MAX * 100:.0f} cm/s：{sum(site) / max(len(site), 1):.1%}（{len(site)} 回合）")
        code = 0 if site and sum(site) / len(site) > 0.3 else 1
        print("任务可解" if code == 0 else "现场带速成功率低于 30%：先看图形界面找原因（去掉 --headless），不要开始训练")
        env.close()
    finally:
        finish(app, code)


if __name__ == "__main__":
    sys.exit(main())
