"""Depth image -> base-frame point cloud and a RANSAC fit of the (near-horizontal) belt plane.

Shared by calibration step5 (belt measurement) and the deployment perception.
"""
from __future__ import annotations

import cv2
import numpy as np


def depth_to_base(depth: np.ndarray, K: np.ndarray, D: np.ndarray, T_base_cam: np.ndarray, step: int = 3,
                  return_pix: bool = False):
    """OpenCV camera (x right, y down, z forward) depth -> base-frame points; optionally their (u, v) pixels."""
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[v, u]
    keep = z > 0
    pix = np.stack([u[keep], v[keep]], axis=1).astype(np.float64)
    raw_pix = pix.copy()
    if np.any(D):
        pix = cv2.undistortPoints(pix.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    z = z[keep]
    pc = np.stack([(pix[:, 0] - K[0, 2]) * z / K[0, 0], (pix[:, 1] - K[1, 2]) * z / K[1, 1], z], axis=1)
    pts = pc @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    return (pts, raw_pix) if return_pix else pts


# 【说明】RANSAC 拟合带面，按闭环 10 Hz 的时间预算写（整次约 3 ms）：
#   · 只在候选点里随机抽的 score_points 个点上取三点、数内点：4000 个点的内点比例足够分出平面好坏，
#     最后的最小二乘精化仍用全部内点，所以平面精度不变
#   · iters 个假设平面一次性算：法向量批量叉乘，"点到每个平面的距离"是一次 (点数 × 假设数) 矩阵乘
#     （原来逐个假设循环，Python 开销就要 25 ms；最早每次数 6 万个点要 85 ms）
#   · 只接受近水平的假设（|法向 z| ≥ 0.95，带面倾角实测 < 1°）
def fit_plane(pts: np.ndarray, z_guess: float, rng=None, iters: int = 400, thresh: float = 0.004,
              score_points: int = 4000):
    rng = rng or np.random.default_rng(0)
    cand = pts[np.abs(pts[:, 2] - z_guess) < 0.15]
    if len(cand) < 100:
        raise RuntimeError("带面初值附近没有点：检查相机是否对准带面、手眼结果是否正确")
    score = cand[rng.choice(len(cand), score_points, replace=False)] if len(cand) > score_points else cand
    a, b, c = score[rng.integers(0, len(score), (3, iters))]          # 每列一个假设：三个点
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n, a = n[ok] / norm[ok, None], a[ok]
    ok = np.abs(n[:, 2]) >= 0.95
    n, a = n[ok], a[ok]
    best = None
    if len(n):
        counts = (np.abs(score @ n.T - np.sum(a * n, axis=1)) < thresh).sum(axis=0)
        k = int(np.argmax(counts))
        best = (n[k], a[k])
    if best is None:
        raise RuntimeError("没有拟合出近水平的带面")
    n, a = best
    inl = cand[np.abs((cand - a) @ n) < thresh]
    A = np.c_[inl[:, 0], inl[:, 1], np.ones(len(inl))]          # 最小二乘精化 z = ax + by + c
    abc = np.linalg.lstsq(A, inl[:, 2], rcond=None)[0]
    on = np.abs(pts[:, 2] - (pts[:, 0] * abc[0] + pts[:, 1] * abc[1] + abc[2])) < thresh
    return abc, on
