"""Chessboard target: object points, corner detection with a canonical corner order, and PnP."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# =============================================================================
# 【说明】标定板：沿用 NERO 工程那块玻璃棋盘格 —— 12×9 个方格，内角点 11×8，方格 20 mm。
#   板坐标系：原点在第一个内角点，x 沿 11 个角点的方向，y 沿 8 个角点的方向，z 指向板内（右手系）。
# 角点顺序的歧义：OpenCV 有时从对角另一端开始编号（相当于板转了 180°），
#   这会让手眼求解里混进"转了 180° 的板"。这里统一规则：第 0 号角点外侧对角的那个方格
#   必须比它旁边的方格暗（即是黑格），否则把角点顺序整体反转。12×9 方格转 180° 后黑白互换，
#   所以这条规则能唯一确定顺序。
# =============================================================================


@dataclass(frozen=True)
class Chessboard:
    cols: int = 11          # 内角点
    rows: int = 8
    square_m: float = 0.02

    def object_points(self) -> np.ndarray:
        pts = np.zeros((self.cols * self.rows, 3))
        pts[:, :2] = np.mgrid[:self.cols, :self.rows].T.reshape(-1, 2) * self.square_m
        return pts

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.cols - 1) * self.square_m / 2, (self.rows - 1) * self.square_m / 2, 0.0])

    def to_dict(self) -> dict:
        return asdict(self)


def _patch_mean(gray: np.ndarray, xy: np.ndarray, r: int = 3) -> float:
    x, y = int(round(xy[0])), int(round(xy[1]))
    h, w = gray.shape
    x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, w), max(y - r, 0), min(y + r + 1, h)
    return float(gray[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else np.nan


def canonical_order(corners: np.ndarray, gray: np.ndarray, board: Chessboard) -> np.ndarray:
    c = corners.reshape(board.rows, board.cols, 2)
    p0, px, py = c[0, 0], c[0, 1], c[1, 0]
    ex, ey = px - p0, py - p0
    outer_diag = _patch_mean(gray, p0 - 0.5 * ex - 0.5 * ey)     # 第 0 号角点外侧对角方格
    outer_side = _patch_mean(gray, p0 + 0.5 * ex - 0.5 * ey)     # 与它相邻的方格（颜色相反）
    if np.isfinite(outer_diag) and np.isfinite(outer_side) and outer_diag > outer_side:
        return corners[::-1].copy()
    return corners


def detect(image_rgb: np.ndarray, board: Chessboard) -> np.ndarray | None:
    """Return (N, 2) sub-pixel corners in canonical order, or None."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (board.cols, board.rows), flags)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    else:                                            # 备用：扇区法检测器，本身就是亚像素精度
        found, corners = cv2.findChessboardCornersSB(gray, (board.cols, board.rows), cv2.CALIB_CB_EXHAUSTIVE)
        if not found:
            return None
    return canonical_order(corners.reshape(-1, 2), gray, board)


def solve_pnp(corners: np.ndarray, board: Chessboard, K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, float]:
    """Board pose in the camera (OpenCV) frame, T_cam_board, and reprojection RMS (px)."""
    obj = board.object_points()
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1, 1, 2), K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP 失败")
    rvec, tvec = cv2.solvePnPRefineLM(obj, corners.reshape(-1, 1, 2), K, D, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, rms


def draw(image_rgb: np.ndarray, corners: np.ndarray | None, board: Chessboard, color=(0, 255, 0)) -> np.ndarray:
    out = image_rgb.copy()
    if corners is not None:
        cv2.drawChessboardCorners(out, (board.cols, board.rows), corners.reshape(-1, 1, 2).astype(np.float32), True)
        cv2.circle(out, tuple(int(v) for v in corners[0]), 8, color, 2)     # 标出第 0 号角点
    return out
