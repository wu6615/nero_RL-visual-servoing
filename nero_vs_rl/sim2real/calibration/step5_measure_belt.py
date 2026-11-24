"""Step 5 - measure the belt (centre line, width, surface height) with the calibrated wrist camera.

  python -m nero_vs_rl.sim2real.calibration.step5_measure_belt --handeye calib_data/<手眼目录> [--apply]
  python -m nero_vs_rl.sim2real.calibration.step5_measure_belt --from calib_data/<带目录> ... --apply [--keep-z]
      （不连设备：把已测好的结果（多次取平均）写入仿真配置；--keep-z 不改带面高度）

先用示教/拖动把机械臂摆到带的正上方（相机朝下、离带面 ~40 cm、整条带宽都在深度画面里），松手静止后运行。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ...sensors.pointcloud import depth_to_base, fit_plane  # noqa: F401（测试也从这里导入）
from ...utils.config import EnvConfig
from .common import SITE_CONFIG, RobotFK, new_session_dir, read_json, write_json
from .sim_bridge import update_site_config

# =============================================================================
# 【说明】带的几何是仿真里物体轨迹和碰撞的基础，现在 belt_center_x、带宽还是估计/占位值。
#   1. 深度图 → 深度相机系点云 → 基座系（关节 → MJCF 正解 gripper_base → 手眼外参 X_depth）
#   2. RANSAC 拟合近水平平面（初值附近 ±15 cm），得到带面高度和倾角
#   3. 带沿基座 Y 方向；在每个 Y 切片里取平面内点的 x 范围，取中位数 → 带的两条边
#   4. 边外侧要有"非带面"的点（护栏更高 / 地面更低）才算看到了真边；否则提示视野没盖住整条带
#   带面高度可与 2026-11-13 独立实测的平面（z = 0.0072x + 0.0131y − 0.1194）对比，顺便检验手眼精度。
# =============================================================================

MEASURED_PLANE_ABC = (0.007199910002951865, 0.013132503329778597, -0.11941365148235557)


def belt_edges(pts: np.ndarray, on: np.ndarray, bin_m: float = 0.02) -> dict:
    p = pts[on]
    ys = np.arange(p[:, 1].min(), p[:, 1].max(), bin_m)
    lo, hi = [], []
    for y0 in ys:
        sl = p[(p[:, 1] >= y0) & (p[:, 1] < y0 + bin_m), 0]
        if len(sl) > 30:
            lo.append(np.percentile(sl, 1))
            hi.append(np.percentile(sl, 99))
    if len(lo) < 3:
        raise RuntimeError("带面点太少，量不出边缘")
    x0, x1 = float(np.median(lo)), float(np.median(hi))
    off = pts[~on]
    seen_lo = bool(np.sum(off[:, 0] < x0 - 0.01) > 50)          # 边外侧确实有别的东西（护栏/地面）
    seen_hi = bool(np.sum(off[:, 0] > x1 + 0.01) > 50)
    return dict(x_min=x0, x_max=x1, center_x=(x0 + x1) / 2, half_width=(x1 - x0) / 2,
                edge_seen_low=seen_lo, edge_seen_high=seen_hi, y_range=[float(ys[0]), float(ys[-1] + bin_m)])


def top_view(pts, on, edges, path: Path, mpp: float = 0.002):
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    W, H = int((hi[0] - lo[0]) / mpp) + 1, int((hi[1] - lo[1]) / mpp) + 1
    img = np.zeros((H, W, 3), np.uint8)
    for sel, col in ((~on, (90, 90, 90)), (on, (0, 200, 0))):
        ij = ((pts[sel, :2] - lo) / mpp).astype(int)
        img[H - 1 - ij[:, 1], ij[:, 0]] = col
    for x in (edges["x_min"], edges["x_max"]):
        c = int((x - lo[0]) / mpp)
        cv2.line(img, (c, 0), (c, H - 1), (0, 0, 255), 1)
    cv2.imwrite(str(path), img)


def apply_belt(results: list[dict], sources: list[str], keep_z: bool, site_config: Path) -> Path:
    """Average one or more belt.json results and write them into the site config."""
    cx = float(np.mean([r["center_x"] for r in results]))
    hw = float(np.mean([r["half_width"] for r in results]))
    z = float(np.mean([r["belt_surface_z_at_center"] for r in results]))
    z_ref = float(np.mean([r["reference_plane_z"] for r in results]))
    updates = dict(belt_center_x=round(cx, 4), belt_half_width=round(hw, 4))
    if not keep_z:
        updates["belt_surface_z"] = round(z, 4)
    prov = dict(step="belt", sessions=sources, n_measurements=len(results),
                center_x_spread_mm=round((max(r["center_x"] for r in results) - min(r["center_x"] for r in results)) * 1000, 1),
                measured_surface_z=round(z, 4), z_diff_vs_reference_mm=round((z - z_ref) * 1000, 1),
                surface_z_written=not keep_z,
                edges_seen=all(r["edge_seen_low"] and r["edge_seen_high"] for r in results))
    if keep_z:
        prov["note"] = "带面高度保留原值（11-13 独立实测、NERO 真机抓取验证过）；本次测得值只作记录"
    path = update_site_config(updates, prov, path=site_config)
    print(f"已写入 {path}：{updates}" + ("（带面高度保持不变）" if keep_z else ""))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="用手眼标定后的腕部相机测带的位置")
    ap.add_argument("--handeye", default=None, help="step2 的手眼目录（含 result.json）；现场测量时必填")
    ap.add_argument("--from", dest="from_dirs", nargs="+", default=None,
                    help="不连设备：用已有的带测量目录（可多个，取平均）")
    ap.add_argument("--keep-z", action="store_true", help="只写中心线和半宽，带面高度保持配置里原值")
    ap.add_argument("--site-config", default=str(SITE_CONFIG))
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--can", default="can0")
    ap.add_argument("--firmware", default="v121")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--apply", action="store_true", help="写入 dc1_site.yaml 的 belt_center_x / belt_half_width / belt_surface_z")
    args = ap.parse_args(argv)
    if args.from_dirs:
        results = [read_json(Path(d) / "belt.json") for d in args.from_dirs]
        for d, r in zip(args.from_dirs, results):
            print(f"{d}：中心 {r['center_x']:.4f}，半宽 {r['half_width']:.4f}，带面 {r['belt_surface_z_at_center']:.4f}"
                  f"（独立实测 {r['reference_plane_z']:.4f}）")
        if not args.apply:
            print("加 --apply 才会写入配置")
            return None
        return apply_belt(results, [str(d) for d in args.from_dirs], args.keep_z, Path(args.site_config))
    if not args.handeye:
        ap.error("现场测量要给 --handeye；只写入已有结果用 --from")
    from ..hardware.dc1_camera import DC1Camera, DC1Config
    from ..hardware.nero_arm import NeroArmReader

    res = read_json(Path(args.handeye) / "result.json")
    if not res.get("accepted"):
        print("⚠ 这次手眼标定没有通过验收，测出的带位置也不可靠")
    X_depth = np.asarray(res["T_gripper_base_depth_cv"])
    di = res["depth_intrinsics"]
    K = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
    D = np.asarray(di.get("dist", [0] * 5))
    fk = RobotFK()
    out = new_session_dir("belt")
    with DC1Camera(DC1Config(serial=args.serial)) as cam, NeroArmReader(args.can, args.firmware) as arm:
        if not arm.stationary(1.0):
            time.sleep(1.2)
        if not arm.stationary(1.0):
            raise SystemExit("机械臂没静止，松手后再运行")
        q = arm.joints()
        depth = np.median(np.stack([cam.read(need_color=False).depth for _ in range(args.frames)]), axis=0)
        if np.max(np.abs(arm.joints() - q)) > 3e-4:
            raise SystemExit("测量期间机械臂动了")
    T_base_cam = fk.body_T(q, "gripper_base") @ X_depth
    pts = depth_to_base(depth, K, D, T_base_cam)
    abc, on = fit_plane(pts, EnvConfig().belt_surface_z)
    edges = belt_edges(pts, on)
    z_c = float(abc[0] * edges["center_x"] + abc[1] * np.mean(edges["y_range"]) + abc[2])
    z_ref = MEASURED_PLANE_ABC[0] * edges["center_x"] + MEASURED_PLANE_ABC[1] * np.mean(edges["y_range"]) + MEASURED_PLANE_ABC[2]
    tilt = float(np.degrees(np.arctan(np.hypot(abc[0], abc[1]))))
    np.save(out / "depth_median.npy", depth)
    top_view(pts, on, edges, out / "belt_top_view.png")
    write_json(out / "belt.json", dict(joints=q, plane_abc=abc, belt_surface_z_at_center=z_c, tilt_deg=tilt,
                                       reference_plane_z=z_ref, handeye_session=args.handeye, **edges))
    print(f"带面高度 {z_c:.4f} m（11-13 独立实测 {z_ref:.4f} m，差 {(z_c - z_ref) * 1000:+.1f} mm），倾角 {tilt:.2f}°")
    print(f"带中心线 x = {edges['center_x']:.4f} m，半宽 {edges['half_width']:.4f} m（x ∈ [{edges['x_min']:.3f}, {edges['x_max']:.3f}]）")
    if not (edges["edge_seen_low"] and edges["edge_seen_high"]):
        print("⚠ 有一侧没看到带外的东西，可能视野没盖住整条带：抬高相机或换位置再测；俯视图见", out / "belt_top_view.png")
    if abs(z_c - z_ref) > 0.01:
        print("⚠ 带面高度与独立实测差 >1 cm：手眼标定或基座系可能有问题")
    if args.apply:
        apply_belt([read_json(out / "belt.json")], [str(out)], args.keep_z, Path(args.site_config))


if __name__ == "__main__":
    main()
