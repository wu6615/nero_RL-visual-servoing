"""Batched SO(3) helpers (torch), same conventions as utils/geometry.py (numpy)."""
from __future__ import annotations

import math

import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    x, y, w = v.unbind(-1)
    return torch.stack([z, -w, y, w, z, -x, -y, x, z], -1).reshape(*v.shape[:-1], 3, 3)


def expmap(w: torch.Tensor) -> torch.Tensor:
    th = w.norm(dim=-1)[..., None, None]
    K = skew(w)
    small = th < 1e-6
    t = torch.where(small, torch.ones_like(th), th)
    A = torch.where(small, torch.ones_like(th), torch.sin(t) / t)
    B = torch.where(small, 0.5 * torch.ones_like(th), (1 - torch.cos(t)) / t ** 2)
    return torch.eye(3, dtype=w.dtype, device=w.device) + A * K + B * (K @ K)


def trace(R: torch.Tensor) -> torch.Tensor:
    return R.diagonal(dim1=-2, dim2=-1).sum(-1)


def vee(R: torch.Tensor) -> torch.Tensor:
    return torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)


def rot_angle(R: torch.Tensor) -> torch.Tensor:
    # 【说明】atan2 形式：float32 下 arccos 在 0° 和 180° 附近误差大（1e-4 量级），atan2 没有这个问题
    return torch.atan2(vee(R).norm(dim=-1) / 2, (trace(R) - 1) / 2)


def logmap(R: torch.Tensor) -> torch.Tensor:
    th = rot_angle(R)
    v = vee(R)
    s = torch.sin(th)
    k = torch.where(th < 1e-4, 0.5 * torch.ones_like(th), th / (2 * torch.where(s.abs() < 1e-9, torch.ones_like(s), s)))
    out = v * k[..., None]
    near_pi = (math.pi - th) < 1e-3
    if near_pi.any():                               # 接近 180°：取对称部分最大列作转轴（同 numpy 版）
        B = (R + torch.eye(3, dtype=R.dtype, device=R.device)) / 2
        i = B.diagonal(dim1=-2, dim2=-1).argmax(-1)
        axis = torch.gather(B, -1, i[..., None, None].expand(*B.shape[:-1], 1)).squeeze(-1)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.where(near_pi[..., None], axis * th[..., None], out)
    return out


def symmetric_rot_dist(R: torch.Tensor) -> torch.Tensor:
    """Angle to a grasp rotation, the 180° flip about the approach axis counting as equal."""
    flip = torch.tensor([-1.0, -1.0, 1.0], dtype=R.dtype, device=R.device)
    return torch.minimum(rot_angle(R), rot_angle(R * flip))          # R @ diag(-1,-1,1) = 各列乘符号


def rot_z(a: torch.Tensor) -> torch.Tensor:
    c, s, z, o = torch.cos(a), torch.sin(a), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*a.shape, 3, 3)


def rot6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def quat_wxyz_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def yaw_to_quat_wxyz(a: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(a)
    return torch.stack([torch.cos(a / 2), z, z, torch.sin(a / 2)], -1)
