"""Step 3 (offline stage) - estimate how much later DC1 frames arrive than the joint readings (camera latency).

  python -m nero_vs_rl.sim2real.deploy.step3_measure_latency --record <录制目录> [--apply]
  python -m nero_vs_rl.sim2real.deploy.step3_measure_latency --latest [--apply]

录制要求：机械臂在动（操作者拖动来回扫，TCP 10～30 cm/s），画面里有物体。物体静止（带停住）最好，
带在匀速走也可以。--apply 把结果写进 dc1_site.yaml 的 camera.latency_s（回放 / 部署取关节时用）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ... import REAL_DATA_DIR, TRAIN_CONFIG
from ...utils.config import load_config
from ...utils.geometry import inv_T, make_T
from .common import JointSource, RobotTwin, load_record, read_depth
from .perception import BeltObjectDetector

# =============================================================================
# 【说明】怎么测延迟：带上的物体要么静止、要么匀速直线运动，在基座系里的轨迹应当是一条直线。
#   机械臂动的时候，如果取关节角的时刻不对（相机帧其实是 L 秒之前拍的），换算到基座系的物体位置
#   会跟着机械臂"晃"。于是：
#     1. 每帧检测一次物体，把物体中心存成相机系坐标（与取哪一刻的关节无关）
#     2. 对每个候选 L：用 t − L 时刻的关节把相机系中心变到基座系，每个物体的轨迹拟合匀速直线，求残差
#     3. 残差最小的 L 就是相机延迟
#   机械臂几乎不动时所有 L 的残差一样，测不出来 → 程序会提示。
# =============================================================================

L_GRID = np.round(np.arange(-0.10, 0.301, 0.01), 3)
MIN_TRACK = 8
MIN_TCP_SPEED = 0.05           # TCP 速度 90 分位低于 5 cm/s 认为机械臂没怎么动


def collect(record: Path, cfg, twin: RobotTwin):
    meta, frames = load_record(record)
    di = meta["camera"]["depth_intrinsics"]
    K = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
    det = BeltObjectDetector(cfg, K, np.asarray(di.get("dist") or [0.0] * 5, float), twin.max_width)
    joints = JointSource(record, frames)
    obs, tcp, ts = [], [], []
    for rec in frames:
        q = joints.at(rec["stamp"])
        twin.set_joints(q)
        p, R = twin.tcp_pose()
        tcp.append(p)
        ts.append(rec["stamp"])
        T_bc = twin.T_base_cam_cv()
        d = det.detect(read_depth(record, rec), T_bc, make_T(R, p))
        if d is not None and not d.truncated:
            obs.append((det.object_id, rec["stamp"], (inv_T(T_bc) @ np.r_[d.center, 1.0])[:3]))
    tcp, ts = np.array(tcp), np.array(ts)
    speed = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-6) if len(ts) > 1 else np.zeros(1)
    return obs, joints, speed


def track_rms(obs, joints: JointSource, twin: RobotTwin, L: float) -> tuple[float, int]:
    res, n = [], 0
    for oid in sorted({o[0] for o in obs}):
        tr = [o for o in obs if o[0] == oid]
        if len(tr) < MIN_TRACK:
            continue
        t = np.array([o[1] for o in tr])
        C = []
        for _, tt, c_cam in tr:
            twin.set_joints(joints.at(tt - L))
            T = twin.T_base_cam_cv()
            C.append(T[:3, :3] @ c_cam + T[:3, 3])
        C = np.array(C)
        A = np.c_[t - t.mean(), np.ones(len(t))]
        coef = np.linalg.lstsq(A, C, rcond=None)[0]
        res.append(np.sum((C - A @ coef) ** 2, axis=1))
        n += 1
    if not res:
        return np.nan, 0
    return float(np.sqrt(np.concatenate(res).mean())), n


def estimate(record: Path, config: str) -> dict:
    cfg, _ = load_config(config)
    twin = RobotTwin(cfg)
    obs, joints, speed = collect(record, cfg, twin)
    curve = [(float(L), *track_rms(obs, joints, twin, L)) for L in L_GRID]
    twin.close()
    ok = [(L, r) for L, r, n in curve if n > 0 and np.isfinite(r)]
    out = dict(record=str(record), joints_source=joints.kind, tcp_speed_p90_cm_s=float(np.percentile(speed, 90) * 100),
               n_tracks=max((n for _, _, n in curve), default=0), curve=[(L, r) for L, r in ok], warnings=[])
    if not ok:
        out["warnings"].append(f"没有足够长的物体轨迹（每个物体至少 {MIN_TRACK} 帧完整检出）")
        return out
    Ls, rs = np.array([x[0] for x in ok]), np.array([x[1] for x in ok])
    k = int(np.argmin(rs))
    out.update(latency_s=float(Ls[k]), rms_mm_best=float(rs[k] * 1000),
               rms_mm_at_zero=float(rs[np.argmin(np.abs(Ls))] * 1000))
    # 置信度：最优处比 L=0 改善多少、曲线是否在两端都抬起来（最优值不在搜索边界上）
    if out["tcp_speed_p90_cm_s"] < MIN_TCP_SPEED * 100:
        out["warnings"].append("机械臂几乎没动，测不出延迟：录制时拖动机械臂来回扫（TCP 10～30 cm/s）")
    if k in (0, len(Ls) - 1):
        out["warnings"].append("最优值落在搜索范围边界上，结果不可信")
    if joints.kind == "frames":
        out["warnings"].append("这段录制没有 100 Hz 关节历史（旧版录制），只能在 10 Hz 逐帧关节之间插值，结果偏粗")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="用机械臂在动的录制估计相机延迟")
    ap.add_argument("--record", default=None)
    ap.add_argument("--latest", action="store_true", help=f"用 {REAL_DATA_DIR} 下最新的录制")
    ap.add_argument("--config", default=str(TRAIN_CONFIG))
    ap.add_argument("--apply", action="store_true", help="写入 dc1_site.yaml 的 camera.latency_s")
    ap.add_argument("--site-config", default=None, help="写入哪个配置文件（默认 dc1_site.yaml）")
    args = ap.parse_args(argv)
    if not args.record and not args.latest:
        ap.error("给 --record 目录，或用 --latest")
    if args.record:
        record = Path(args.record)
    else:
        from .step2_replay import latest_record
        record = latest_record()
    print(f"录制: {record}")
    r = estimate(record, args.config)
    print(f"关节来源 {r['joints_source']}，TCP 速度 90 分位 {r['tcp_speed_p90_cm_s']:.1f} cm/s，物体轨迹 {r['n_tracks']} 条")
    for L, rms in r["curve"][::3]:
        print(f"  L = {L:+.2f} s   轨迹残差 {rms * 1000:5.1f} mm")
    if "latency_s" in r:
        print(f"相机延迟 ≈ {r['latency_s']:.2f} s（残差 {r['rms_mm_best']:.1f} mm，不补偿时 {r['rms_mm_at_zero']:.1f} mm）")
    for w in r["warnings"]:
        print("⚠", w)
    if args.apply:
        if "latency_s" not in r or r["warnings"]:
            raise SystemExit("结果不可靠，没有写入（先处理上面的警告）")
        from ..calibration.common import SITE_CONFIG
        from ..calibration.sim_bridge import update_site_config
        path = update_site_config({"camera": {"latency_s": round(r["latency_s"], 3)}},
                                  dict(step="camera_latency", session=str(record), latency_s=round(r["latency_s"], 3),
                                       rms_mm_best=round(r["rms_mm_best"], 1), rms_mm_at_zero=round(r["rms_mm_at_zero"], 1)),
                                  path=Path(args.site_config) if args.site_config else SITE_CONFIG)
        print(f"已写入 {path}：camera.latency_s = {r['latency_s']:.3f}")
    return r


if __name__ == "__main__":
    main()
