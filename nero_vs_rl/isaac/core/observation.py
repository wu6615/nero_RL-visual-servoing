"""Batched observation dict: port of envs/observation.py::assemble_obs (same keys, shapes and slot order)."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from .so3 import rot6d, symmetric_rot_dist


def observation_space(N: int, K: int) -> spaces.Dict:
    """Identical to DynamicGraspEnv.observation_space (checked by tests) so policies move between simulators."""
    box = lambda lo, hi, s: spaces.Box(lo, hi, s, np.float32)
    inf = np.inf
    return spaces.Dict({
        "tcp_pose": box(-inf, inf, (9,)), "gripper_pos": box(0.0, 1.0, (1,)), "action": box(-1.0, 1.0, (7,)),
        "grasp_vis": box(0.0, 1.0, (2,)), "origin_gripper_pts": box(-inf, inf, (K, 3)),
        "gripper_pts_diff": box(-inf, inf, (N, K, 3)), "grasp_scores": box(0.0, 1.0, (N,)),
    })


def action_space() -> spaces.Box:
    return spaces.Box(-1.0, 1.0, (7,), np.float32)


def assemble_obs(p, R, Gw, scores, ok, visible, age, grip_norm, prev_action, pts, N: int) -> dict[str, torch.Tensor]:
    """p (B,3) R (B,3,3) Gw (B,G,4,4) ok (B,G) pts (B,K,3)."""
    B, G = ok.shape
    Rt = R.transpose(1, 2)
    Re = Rt[:, None] @ Gw[..., :3, :3]
    te = (Gw[..., :3, 3] - p[:, None]) @ R
    dist = (te.norm(dim=-1) + 0.1 * symmetric_rot_dist(Re)).masked_fill(~ok, float("inf"))
    order = dist.argsort(dim=1, stable=True)[:, :N]
    n = order.shape[1]
    take = lambda x: torch.gather(x, 1, order.reshape(B, n, *([1] * (x.dim() - 2))).expand(B, n, *x.shape[2:]))
    Re, te, sc, ok = take(Re), take(te), take(scores), take(ok)
    diff = torch.einsum("bkj,bnij->bnki", pts, Re) + te[:, :, None]
    diff = diff * ok[..., None, None]
    sc = sc * ok
    if n < N:
        diff = torch.cat([diff, diff.new_zeros(B, N - n, *diff.shape[2:])], 1)
        sc = torch.cat([sc, sc.new_zeros(B, N - n)], 1)
    f = lambda x: x.to(torch.float32)
    return {"tcp_pose": f(torch.cat([p, rot6d(R)], 1)), "gripper_pos": f(grip_norm[:, None].clamp(0, 1)), "action": f(prev_action),
            "grasp_vis": f(torch.stack([visible.to(p.dtype), age], 1)), "origin_gripper_pts": f(pts),
            "gripper_pts_diff": f(diff), "grasp_scores": f(sc)}
