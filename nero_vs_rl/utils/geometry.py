"""Small SE(3)/SO(3) helpers (numpy only) shared by sim, env and scripted policy."""
from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, p = T[..., :3, :3], T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, p)
    out[..., 3, 3] = 1.0
    return out


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expmap(w: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.eye(3) + skew(w)
    k = skew(w / theta)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def logmap(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector, robust near 0 and pi."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-6:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-4:
        # 接近 180°：取对称部分最大列作为转轴
        B = (R + np.eye(3)) / 2
        axis = B[:, int(np.argmax(np.diag(B)))]
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * theta / (2 * np.sin(theta))


def rot_angle(R: np.ndarray) -> np.ndarray:
    """Geodesic angle of (batched) rotation matrices."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))


RZ_PI = np.diag([-1.0, -1.0, 1.0])  # 绕 TCP z（接近方向）转 180°：二指夹爪对称


def symmetric_grasp_rot_dist(R_rel: np.ndarray) -> np.ndarray:
    """Angle (rad) to a grasp rotation, treating the 180° flip about the approach axis as equal."""
    return np.minimum(rot_angle(R_rel), rot_angle(R_rel @ RZ_PI))


def rot6d(R: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation (first two columns)."""
    return np.concatenate([R[:, 0], R[:, 1]])


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    return expmap(axis / np.linalg.norm(axis) * angle)
