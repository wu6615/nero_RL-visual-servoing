"""Step 4 - measure the DC1 near-range depth blind zone (min_depth_m used by the sim's visibility model).

  python -m nero_vs_rl.sim2real.calibration.step4_measure_blind_zone [--apply]

做法：拿一块哑光平板（纸板/标定板背面）正对相机，从 ~40 cm 慢慢移近，直到画面中央深度变黑；
来回做 2~3 次，按 q 结束。程序取"中央区域仍有效时的最近距离"作为盲区。
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from .common import new_session_dir, write_json
from .sim_bridge import update_site_config

# =============================================================================
# 【说明】为什么要测：眼在手上最后几厘米靠记忆外推，盲区越大，策略"盲抓"的距离越长。
#   仿真里 camera.min_depth_m 决定抓取何时"看不见"，必须和真机一致，否则训练出的策略到真机会失灵。
# 判定：中央 ROI（画面 15%×15%）有效像素比例 ≥ 80% 记为"看得到"，取这些帧里中位深度的
#   1% 分位数作为最近有效距离，写入时再加 1 cm 余量。
# =============================================================================

VALID_RATIO = 0.8


def roi_stats(depth: np.ndarray, frac: float = 0.15) -> tuple[float, float]:
    h, w = depth.shape
    r = depth[int(h * (0.5 - frac / 2)):int(h * (0.5 + frac / 2)), int(w * (0.5 - frac / 2)):int(w * (0.5 + frac / 2))]
    valid = r > 0
    ratio = float(valid.mean())
    return (float(np.median(r[valid])) if valid.any() else 0.0), ratio


def estimate_min_depth(records: list[tuple[float, float]]) -> float | None:
    """records = [(median_depth_m, valid_ratio)]; nearest distance at which the centre is still measured."""
    good = np.array([d for d, r in records if r >= VALID_RATIO and d > 0])
    return float(np.percentile(good, 1)) if len(good) >= 5 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="测 DC1 近距盲区")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--apply", action="store_true", help="写入 dc1_site.yaml 的 camera.min_depth_m")
    args = ap.parse_args(argv)
    from ..hardware.dc1_camera import DC1Camera, DC1Config

    out = new_session_dir("blindzone")
    records = []
    win = "step4 blind zone (q to finish)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    with DC1Camera(DC1Config(serial=args.serial)) as cam:
        while True:
            f = cam.read(need_color=False)
            d, ratio = roi_stats(f.depth)
            records.append((d, ratio))
            vis = cv2.applyColorMap(cv2.convertScaleAbs(f.depth, alpha=255 / 1.0), cv2.COLORMAP_JET)
            vis[f.depth <= 0] = 0
            h, w = f.depth.shape
            cv2.rectangle(vis, (int(w * .425), int(h * .425)), (int(w * .575), int(h * .575)), (255, 255, 255), 2)
            est = estimate_min_depth(records)
            for i, s in enumerate([f"center depth {d * 100:5.1f} cm  valid {ratio * 100:3.0f}%",
                                   f"nearest valid so far: {est * 100:.1f} cm" if est else "nearest valid: -"]):
                cv2.putText(vis, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    est = estimate_min_depth(records)
    write_json(out / "blind_zone.json", dict(records=records, min_valid_depth_m=est, valid_ratio=VALID_RATIO,
                                             created=time.strftime("%F %T")))
    if est is None:
        raise SystemExit("有效帧太少，没测出来：确认平板挡住了画面中央、从远处慢慢移近")
    value = round(est + 0.01, 3)
    print(f"中央仍有效的最近距离 ≈ {est * 100:.1f} cm → 建议 camera.min_depth_m = {value}（含 1 cm 余量）；数据 {out}")
    if args.apply:
        path = update_site_config({"camera": {"min_depth_m": value}},
                                  dict(step="blind_zone", session=str(out), measured_m=round(est, 4)))
        print(f"已写入 {path}")


if __name__ == "__main__":
    main()
