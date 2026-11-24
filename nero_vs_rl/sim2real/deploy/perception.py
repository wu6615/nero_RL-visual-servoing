"""Real-robot perception: DC1 depth -> belt plane -> the object on the belt -> box/cylinder pose and grasps.

Replaces the simulated "noisy ground-truth grasps" of ``DynamicGraspEnv._observe_grasps``.  The output
feeds the same ``GraspTracker`` / ``assemble_obs`` as the simulation, so the policy sees the same kind of input.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from ...grasp_detection import analytic_grasps as gr
from ...sensors import camera as cam
from ...sensors.pointcloud import depth_to_base, fit_plane
from ...utils.config import EnvConfig
from ...utils.geometry import make_T, rot_z

# =============================================================================
# 【说明】一帧深度图里找带上的物体（只针对训练用的长方体 / 圆柱，平放在带面上）：
#   1. 深度 → 基座系点云（关节正解 × 手眼外参，畸变按 DC1 出厂参数去掉）
#   2. RANSAC 拟合带面（近水平，初值 belt_surface_z ±15 cm）；拟合失败沿用上一帧的平面
#      ⇒ 不依赖配置里带面高度的毫米级精度（step5 测得 −0.112 vs 配置 −0.12）
#   3. 候选点：在带宽范围内、高出带面 12 mm～15 cm；去掉 TCP 附近夹爪自身的点
#   4. 俯视 5 mm 栅格做连通域 → 选点数最多（或离上一帧最近）的一团
#   5. 俯视轮廓：高于"顶高 × (1 − FOOTPRINT_FRAC)"的点 → cv2.minAreaRect 得到俯视矩形：中心、边长、朝向
#      高度 = 最高处（95 分位）相对带面的高度；圆柱 / 长方体按"凸包面积 / 外接矩形面积"区分（圆 ≈ 0.785，矩形 ≈ 1）
#      为什么不用"最高那一层"：2026-11-22 真机录制的长方体（46×82×38 mm）顶上有 20 mm 高的圆形凸台，
#      只看最高层会把凸台当成直径 3～4 cm 的圆柱，手指按它去抓会插到底座上；用上半部分轮廓才量出底座 4.6×8.3 cm。
#      训练用的长方体 / 圆柱是直上直下的，两种做法结果一样。
#   6. 尺寸、类型对最近若干帧取中位数 / 多数票（真机单帧轮廓噪声约 1 cm，中位数后误差 < 5 mm）
#   7. 物体系 → analytic_grasps 生成抓取（与仿真同一个函数）；可见性同仿真：抓取可见点要在画面内、深度在量程内
#   物体轮廓被画面边缘截断、或者窄得不像训练物体（被手指挡得只剩一条边）时尺寸会偏小 → 这一帧不用（当作没看到，交给记忆外推）
# =============================================================================

MIN_HEIGHT = 0.012           # 高出带面多少才算物体（训练物体最矮 3 cm）
MAX_HEIGHT = 0.15
GRID = 0.005                 # 俯视连通域栅格
MIN_CLUSTER_POINTS = 40
FOOTPRINT_FRAC = 0.5          # 俯视轮廓用高于 (1-此值)×顶高 的点（见下方说明）
CYL_FILL_MAX = 0.88          # 凸包 / 外接矩形 面积比低于它且接近正方形 → 圆柱
BORDER_PX = 4
MIN_FOOTPRINT_M = 0.02       # 俯视轮廓短边小于它 → 和"被画面截断"一样，这一帧尺寸不用
LOST_FRAMES = 10             # 连续这么多帧没检出 → 当作物体已经离开
NEW_OBJECT_JUMP = 0.10       # 检出位置一帧跳这么远 → 换了一个物体


@dataclass
class Detection:
    obj_type: str                  # box | cylinder
    center: np.ndarray             # 物体中心（基座系）
    yaw: float
    half: np.ndarray               # box: (hx, hy, hz)；cylinder: (r, r, h)
    n_points: int
    truncated: bool                # 碰到画面边缘
    fill: float
    plane_abc: np.ndarray
    raw_half: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 本帧未平滑的尺寸

    @property
    def T_wo(self) -> np.ndarray:
        return make_T(rot_z(self.yaw), self.center)


class BeltObjectDetector:
    def __init__(self, cfg: EnvConfig, K: np.ndarray, D: np.ndarray, max_width: float, pixel_step: int = 2,
                 smooth_frames: int = 10):
        self.cfg, self.K, self.D = cfg, np.asarray(K, float), np.asarray(D, float)
        self.max_width, self.step = max_width, pixel_step
        self.plane = np.array([0.0, 0.0, cfg.belt_surface_z])
        self._sizes: deque = deque(maxlen=smooth_frames)
        self._types: deque = deque(maxlen=smooth_frames)
        self.prev_center = None
        self.object_id, self.new_object, self._missed = -1, False, 0
        self.debug: dict = {}

    def reset(self):
        self.plane = np.array([0.0, 0.0, self.cfg.belt_surface_z])
        self._forget()

    def _forget(self):
        self._sizes.clear()
        self._types.clear()
        self.prev_center = None
        self._missed = 0

    def detect(self, depth: np.ndarray, T_base_cam: np.ndarray, T_base_tcp: np.ndarray) -> Detection | None:
        # 【说明】一次录制 / 一次运行里会有多个物体先后经过：连续 LOST_FRAMES 帧没检出，或者检出的位置
        #   离上一帧跳了 NEW_OBJECT_JUMP 以上 → 认为换了物体，清掉尺寸 / 类型历史，new_object=True
        #   （调用方据此重置 GraspTracker，免得把两个物体的位置混进同一个测速窗口）
        self.new_object = False
        det = self._detect(depth, T_base_cam, T_base_tcp)
        if det is None:
            self._missed += 1
            if self._missed >= LOST_FRAMES and self.prev_center is not None:
                self._forget()
            return None
        self._missed = 0
        return det

    def plane_z(self, xy: np.ndarray) -> np.ndarray:
        a, b, c = self.plane
        return a * xy[..., 0] + b * xy[..., 1] + c

    def _detect(self, depth: np.ndarray, T_base_cam: np.ndarray, T_base_tcp: np.ndarray) -> Detection | None:
        c = self.cfg
        self.debug = {}
        pts, pix = depth_to_base(depth, self.K, self.D, T_base_cam, step=self.step, return_pix=True)
        if len(pts) < 200:
            self.debug["why"] = "有效深度点太少（盲区内或镜头被挡）"
            return None
        try:
            self.plane, _ = fit_plane(pts, c.belt_surface_z, iters=150)
        except RuntimeError:
            pass                                             # 看不到带面：沿用上一帧平面
        h = pts[:, 2] - self.plane_z(pts[:, :2])
        in_belt = np.abs(pts[:, 0] - c.belt_center_x) < c.belt_half_width - 0.005
        p_tcp = (pts - T_base_tcp[:3, 3]) @ T_base_tcp[:3, :3]
        gripper = (np.abs(p_tcp[:, 0]) < 0.035) & (np.abs(p_tcp[:, 1]) < 0.075) & (p_tcp[:, 2] < 0.03)
        sel = in_belt & (h > MIN_HEIGHT) & (h < MAX_HEIGHT) & ~gripper
        self.debug.update(plane=self.plane.copy(), n_candidates=int(sel.sum()))
        if sel.sum() < MIN_CLUSTER_POINTS:
            self.debug["why"] = "带面上方没有物体点"
            return None
        P, H, UV = pts[sel], h[sel], pix[sel]

        # 俯视栅格连通域
        ij = np.floor((P[:, :2] - P[:, :2].min(0)) / GRID).astype(int)
        occ = np.zeros(ij.max(0) + 1, dtype=bool)
        occ[ij[:, 0], ij[:, 1]] = True
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        pl = lab[ij[:, 0], ij[:, 1]]
        counts = np.bincount(pl, minlength=n + 1)
        counts[0] = 0
        cand = [k for k in range(1, n + 1) if counts[k] >= MIN_CLUSTER_POINTS]
        if not cand:
            self.debug["why"] = "物体点太散"
            return None
        if self.prev_center is not None:
            cent = {k: P[pl == k, :2].mean(0) for k in cand}
            near = [k for k in cand if np.linalg.norm(cent[k] - self.prev_center[:2]) < 0.06]
            cand = near or cand
        k = max(cand, key=lambda j: counts[j])
        P, H, UV = P[pl == k], H[pl == k], UV[pl == k]

        top = float(np.percentile(H, 95))
        on_top = H > max(MIN_HEIGHT, (1 - FOOTPRINT_FRAC) * top)
        if on_top.sum() < 10:
            on_top[:] = True
        T, UVt = P[on_top, :2], UV[on_top]
        xy = T.astype(np.float32)
        rect = cv2.minAreaRect(xy)
        box = cv2.boxPoints(rect).astype(float)
        e0, e1 = box[1] - box[0], box[2] - box[1]
        L0, L1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
        yaw = float(np.arctan2(e0[1], e0[0]))
        hull_area = float(cv2.contourArea(cv2.convexHull(xy)))
        fill = hull_area / max(L0 * L1, 1e-9)
        square = abs(L0 - L1) / max(L0, L1, 1e-9) < 0.2
        obj_type = "cylinder" if (fill < CYL_FILL_MAX and square) else "box"
        H_full = top
        center_xy = np.asarray(rect[0], float)
        center = np.array([*center_xy, float(self.plane_z(center_xy)) + H_full / 2])
        raw = (np.array([(L0 + L1) / 4, (L0 + L1) / 4, H_full / 2]) if obj_type == "cylinder"
               else np.array([L0 / 2, L1 / 2, H_full / 2]))
        W, Hh = depth.shape[1], depth.shape[0]
        # 量尺寸用的那部分点碰到画面边缘时，俯视尺寸不可信（下半部分侧面被截断不影响）
        truncated = bool(np.any((UVt[:, 0] < BORDER_PX) | (UVt[:, 0] > W - 1 - BORDER_PX)
                                | (UVt[:, 1] < BORDER_PX) | (UVt[:, 1] > Hh - 1 - BORDER_PX)))
        # 俯视轮廓比训练里最窄的物体（3 cm）还窄得多：多半只看到被手指挡剩的一条边，尺寸同样不可信
        #   （v6 起物体会贴着护栏走，仿真录制里出现过把圆柱量成 2.4 × 0.4 cm 的长条、判成长方体）
        truncated |= min(L0, L1) < MIN_FOOTPRINT_M
        if self.prev_center is None or np.linalg.norm(center[:2] - self.prev_center[:2]) > NEW_OBJECT_JUMP:
            self._forget()
            self.object_id += 1
            self.new_object = True
        self.prev_center = center
        if not truncated:
            # 长方体两条边按"短边在前"排序后再平滑，避免 minAreaRect 边序跳变
            self._sizes.append(np.sort(raw[:2]).tolist() + [raw[2]])
            self._types.append(obj_type)
        if self._types:
            obj_type = Counter(self._types).most_common(1)[0][0]
            med = np.median(np.array(self._sizes), axis=0)
            half = (np.array([med[:2].mean(), med[:2].mean(), med[2]]) if obj_type == "cylinder"
                    else (med if raw[0] <= raw[1] else med[[1, 0, 2]]))
        else:
            half = raw
        return Detection(obj_type, center, yaw, half, len(P), truncated, fill, self.plane.copy(), raw)

    def grasps(self, det: Detection, max_tilt_deg: float) -> gr.GraspSet:
        if det.obj_type == "cylinder":
            return gr.cylinder_grasps(float(det.half[0]), float(det.half[2]), self.max_width, max_tilt_deg)
        return gr.box_grasps(det.half, self.max_width, max_tilt_deg)


def observe(detector: BeltObjectDetector, depth: np.ndarray, T_base_cam_cv: np.ndarray, T_base_tcp: np.ndarray,
            cam_pos_mj: np.ndarray, cam_mat_mj: np.ndarray, cfg: EnvConfig):
    """One frame -> (detection | None, grasps_world | None, scores | None, why_not).

    Same visibility rule as the sim (minus the ray-cast occlusion, which the real depth already contains):
    a grasp is visible when its visibility point projects inside the image and within the depth range.
    """
    det = detector.detect(depth, T_base_cam_cv, T_base_tcp)
    if det is None:
        return None, None, None, detector.debug.get("why", "")
    if det.truncated:
        return det, None, None, "物体被画面边缘截断（本帧不用）"
    gs = detector.grasps(det, cfg.grasp_obs.max_tilt_deg)
    if len(gs) == 0:
        return det, None, None, f"物体太宽（{2 * det.half[0] * 100:.1f}×{2 * det.half[1] * 100:.1f} cm），超出夹爪"
    T_wo = det.T_wo
    vis_world = (T_wo[:3, :3] @ gs.vis_obj.T).T + T_wo[:3, 3]
    uv, d = cam.project(vis_world, cam_pos_mj, cam_mat_mj, cfg.camera)
    m, cc = cfg.camera.image_margin_px, cfg.camera
    mask = ((d >= cc.min_depth_m) & (d <= cc.max_depth_m) & (uv[:, 0] >= m) & (uv[:, 0] <= cc.width - m)
            & (uv[:, 1] >= m) & (uv[:, 1] <= cc.height - m))
    if not mask.any():
        return det, None, None, "抓取点都在盲区或画面外"
    Gw = np.einsum("ij,njk->nik", T_wo, gs.T_obj)[mask]
    return det, Gw, gs.scores[mask], ""


def size_in_training_range(det: Detection, cfg: EnvConfig) -> list[str]:
    """Warnings when the detected object is outside what the policy was trained on."""
    oc, w = cfg.objects, []
    if det.obj_type == "box":
        hx, hy = np.sort(det.half[:2])
        lo_x, hi_x = oc.box_half_x
        lo_y, hi_y = oc.box_half_y
        if not (lo_x - 0.005 <= hx <= hi_x + 0.005):
            w.append(f"短边 {2 * hx * 100:.1f} cm 超出训练范围 {2 * lo_x * 100:.0f}～{2 * hi_x * 100:.0f} cm")
        if not (min(lo_x, lo_y) - 0.005 <= hy <= max(hi_x, hi_y) + 0.005):
            w.append(f"长边 {2 * hy * 100:.1f} cm 超出训练范围")
        lo_z, hi_z = oc.box_half_z
    else:
        lo, hi = oc.cyl_radius
        if not (lo - 0.005 <= det.half[0] <= hi + 0.005):
            w.append(f"直径 {2 * det.half[0] * 100:.1f} cm 超出训练范围 {2 * lo * 100:.0f}～{2 * hi * 100:.0f} cm")
        lo_z, hi_z = oc.cyl_half_height
    if not (lo_z - 0.005 <= det.half[2] <= hi_z + 0.005):
        w.append(f"高度 {2 * det.half[2] * 100:.1f} cm 超出训练范围 {2 * lo_z * 100:.0f}～{2 * hi_z * 100:.0f} cm")
    return w


