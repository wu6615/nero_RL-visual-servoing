"""Eye-in-hand calibration math: AX = XB (OpenCV), consistency checks, reprojection refinement.

Frames (all metres, cameras in OpenCV axes):
  T_base_grip   gripper_base pose in the robot base (MJCF forward kinematics of measured joints)
  T_cam_board   chessboard pose in the camera (PnP)
  X = T_grip_cam   unknown, constant (camera rigidly mounted on the gripper)
  B = T_base_board unknown, constant (board fixed in the lab)
  For every sample i:  T_base_grip_i · X · T_cam_board_i = B
Samples may come from several capture sessions (``Sample.group``); each group
gets its own board pose B_g, so a board nudged between sessions does no harm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .board import Chessboard

METHODS = {
    "park": cv2.CALIB_HAND_EYE_PARK,
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
}


@dataclass
class Sample:
    T_base_grip: np.ndarray
    T_cam_board: np.ndarray
    corners: np.ndarray            # (N, 2) 检测到的角点
    pnp_rms: float = 0.0
    sample_id: int = 0
    extra: dict = field(default_factory=dict)
    group: int = 0                 # 采集会话编号：同一会话里标定板不动，会话之间可能被碰过


# ---------------------------------------------------------------- SE(3) 小工具
def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def mean_pose(Ts: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def _vec_to_T(v: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(v[:3])[0]
    T[:3, 3] = v[3:]
    return T


def _T_to_vec(T: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


# ---------------------------------------------------------------- 求解与评估
def solve(samples: list[Sample], method: str = "park") -> np.ndarray:
    """X = T_grip_cam.  Eye-in-hand: OpenCV takes gripper->base and target->camera directly."""
    R, t = cv2.calibrateHandEye(
        [s.T_base_grip[:3, :3] for s in samples], [s.T_base_grip[:3, 3] for s in samples],
        [s.T_cam_board[:3, :3] for s in samples], [s.T_cam_board[:3, 3] for s in samples],
        method=METHODS[method])
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.ravel()
    return X


def board_in_base(samples: list[Sample], X: np.ndarray) -> list[np.ndarray]:
    return [s.T_base_grip @ X @ s.T_cam_board for s in samples]


def _groups(samples: list[Sample]) -> list[int]:
    return sorted({s.group for s in samples})


def consistency(samples: list[Sample], X: np.ndarray) -> dict:
    """How well samples agree on the board pose in the base frame (one B per session group).

    这个离散度主要反映"机械臂名义运动学 vs 真实位姿"的误差（关节弹性、背隙、零位），
    样本越多也不会变小；X 本身准不准看 stability()。
    """
    Bs = board_in_base(samples, X)
    B_g = {g: mean_pose([Bi for Bi, s in zip(Bs, samples) if s.group == g]) for g in _groups(samples)}
    dt = np.array([np.linalg.norm(Bi[:3, 3] - B_g[s.group][:3, 3]) for Bi, s in zip(Bs, samples)])
    dr = np.array([rot_deg(B_g[s.group][:3, :3].T @ Bi[:3, :3]) for Bi, s in zip(Bs, samples)])
    return dict(B=B_g[min(B_g)], B_groups=B_g, trans_rms_m=float(np.sqrt(np.mean(dt ** 2))),
                trans_max_m=float(dt.max()), rot_rms_deg=float(np.sqrt(np.mean(dr ** 2))),
                rot_max_deg=float(dr.max()), per_sample_trans_m=dt.tolist(), per_sample_rot_deg=dr.tolist())


def project_chain(s: Sample, X: np.ndarray, B: np.ndarray, board: Chessboard, K, D) -> np.ndarray:
    T_cb = inv(X) @ inv(s.T_base_grip) @ B
    rvec = cv2.Rodrigues(T_cb[:3, :3])[0]
    proj, _ = cv2.projectPoints(board.object_points(), rvec, T_cb[:3, 3], K, D)
    return proj.reshape(-1, 2)


def reprojection(samples: list[Sample], X, B, board: Chessboard, K, D) -> np.ndarray:
    """Per-sample RMS (px) of corners predicted through robot FK + X + B vs detected corners.

    B: one board pose, or {group: board pose}.
    """
    Bg = B if isinstance(B, dict) else {g: B for g in _groups(samples)}
    return np.array([np.sqrt(np.mean(np.sum((project_chain(s, X, Bg[s.group], board, K, D) - s.corners) ** 2, axis=1)))
                     for s in samples])


def refine(samples: list[Sample], X0, B0, board: Chessboard, K, D) -> tuple[np.ndarray, dict]:
    """Jointly refine X and every group's B by minimising pixel reprojection (Huber loss)."""
    obj = board.object_points()
    groups = _groups(samples)
    B0 = B0 if isinstance(B0, dict) else {g: B0 for g in groups}
    gi = {g: i for i, g in enumerate(groups)}

    def residual(p):
        X = _vec_to_T(p[:6])
        Bs = [_vec_to_T(p[6 + 6 * i:12 + 6 * i]) for i in range(len(groups))]
        res = []
        for s in samples:
            T_cb = inv(X) @ inv(s.T_base_grip) @ Bs[gi[s.group]]
            proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, D)
            res.append((proj.reshape(-1, 2) - s.corners).ravel())
        return np.concatenate(res)

    p0 = np.concatenate([_T_to_vec(X0)] + [_T_to_vec(B0[g]) for g in groups])
    sol = least_squares(residual, p0, loss="huber", f_scale=1.0, x_scale="jac", max_nfev=200)
    return _vec_to_T(sol.x[:6]), {g: _vec_to_T(sol.x[6 + 6 * i:12 + 6 * i]) for i, g in enumerate(groups)}


def stability(samples: list[Sample], X: np.ndarray, B: dict, board: Chessboard, K, D, n_folds: int = 3) -> dict:
    """How much X moves when part of the data is left out (the real accuracy check for X).

    多个会话：每次去掉一整个会话重算；只有一个会话：按顺序交错分成 n_folds 份，每次去掉一份。
    """
    groups = _groups(samples)
    if len(groups) >= 3:
        folds = [[s for s in samples if s.group != g] for g in groups]
        how = "leave_one_session_out"
    else:
        folds = [[s for i, s in enumerate(samples) if i % n_folds != k] for k in range(n_folds)]
        how = f"leave_one_of_{n_folds}_folds_out"
    dts, drs = [], []
    for sub in folds:
        Xk, _ = refine(sub, X, {g: B[g] for g in _groups(sub)}, board, K, D)
        dts.append(float(np.linalg.norm(Xk[:3, 3] - X[:3, 3])))
        drs.append(rot_deg(X[:3, :3].T @ Xk[:3, :3]))
    return dict(method=how, trans_max_m=max(dts), rot_max_deg=max(drs), per_fold_trans_m=dts, per_fold_rot_deg=drs)


def motion_quality(samples: list[Sample]) -> dict:
    """Hand-eye needs large rotations about different axes; report span and axis diversity."""
    Rs = [s.T_base_grip[:3, :3] for s in samples]
    rel = [Ra.T @ Rb for Ra, Rb in combinations(Rs, 2)]
    angles = [rot_deg(R) for R in rel]
    vecs = np.array([cv2.Rodrigues(R)[0].ravel() for R in rel if rot_deg(R) > 2])
    sv = (np.linalg.svd(vecs / np.linalg.norm(vecs, axis=1, keepdims=True), compute_uv=False) / np.sqrt(len(vecs))
          if len(vecs) >= 3 else np.zeros(3))
    return dict(max_rel_rot_deg=float(max(angles) if angles else 0.0),
                axis_diversity=[float(v) for v in sv],          # 三个值都不小（>0.3）说明绕三个轴都转过
                n_samples=len(samples))


def calibrate(samples: list[Sample], board: Chessboard, K, D, methods=("park", "tsai", "daniilidis"),
              with_stability: bool = True) -> dict:
    """Closed-form methods for a start, joint reprojection refinement, then consistency + stability."""
    report = {"methods": {}}
    for m in methods:
        X = solve(samples, m)        # 闭式解只用作初值；会话间板若被碰过几毫米，影响可忽略，精化时按会话分开
        c = consistency(samples, X)
        report["methods"][m] = dict(X=X, trans_max_m=c["trans_max_m"], rot_max_deg=c["rot_max_deg"])
    best = min(report["methods"], key=lambda m: report["methods"][m]["trans_max_m"])
    X0 = report["methods"][best]["X"]
    X, B_g = refine(samples, X0, consistency(samples, X0)["B_groups"], board, K, D)
    c = consistency(samples, X)
    reproj = reprojection(samples, X, B_g, board, K, D)
    spread = [np.linalg.norm(report["methods"][m]["X"][:3, 3] - X[:3, 3]) for m in methods]
    report.update(best_closed_form=best, X=X, B=B_g[min(B_g)], B_groups=B_g, consistency=c,
                  reprojection_px=reproj.tolist(), reprojection_rms_px=float(np.sqrt(np.mean(reproj ** 2))),
                  method_spread_m=float(max(spread)), motion=motion_quality(samples), n_groups=len(B_g))
    if with_stability:
        report["stability"] = stability(samples, X, B_g, board, K, D)
    return report
