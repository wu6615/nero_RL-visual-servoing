"""Step 2 - solve the eye-in-hand calibration, judge quality, optionally write it into the sim config.

  python -m nero_vs_rl.sim2real.calibration.step2_solve_handeye                 # 默认取最新一次采样
  python -m nero_vs_rl.sim2real.calibration.step2_solve_handeye --session calib_data/<目录> --apply
  python -m nero_vs_rl.sim2real.calibration.step2_solve_handeye --all-sessions --apply   # 合并全部采样（推荐）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ..hardware.dc1_camera import Intrinsics
from .board import Chessboard, solve_pnp
from .common import (CALIB_DATA_DIR, GROSS_BOARD_SPREAD_DEG, GROSS_BOARD_SPREAD_M, MAX_BOARD_SPREAD_DEG,
                     MAX_BOARD_SPREAD_M, MAX_PNP_RMS_PX, MAX_X_INSTABILITY_DEG, MAX_X_INSTABILITY_M,
                     MIN_ROTATION_SPAN_DEG, MIN_SAMPLES, SITE_CONFIG, append_jsonl, read_json, read_jsonl, write_json)
from .handeye import Sample, calibrate, inv, mean_pose, rot_deg, stability
from .sim_bridge import camera_overrides, update_site_config

# =============================================================================
# 【说明】求解流程：
#   1. 读样本 → 用彩色内参（出厂，或 --intrinsics board 用这批图自标）重新 PnP，丢掉 PnP 误差 > 1 px 的
#   2. Park / Tsai / Daniilidis 三种闭式解，取一致性最好的做初值，再以"全部角点重投影误差"联合优化 X 和 B
#   3. 自动剔除离群样本（重投影 > max(3×中位数, 2 px)），最多剔 20%
#   4. 判定：样本数、旋转跨度、X 的稳定性（去掉一个会话 / 三分之一数据重算，X 变化 ≤3 mm / ≤0.5°）；
#      各样本推算的板位姿偏差反映机械臂自身的绝对精度，只在出现粗差（>15 mm / 3°）时判失败
#   多会话合并（--session A B C 或 --all-sessions）：每个会话一个板位姿（会话之间板被碰过也没关系），
#      合并后的样本写到新目录 calib_data/<时间>_handeye_merged/，图像仍引用原目录
#   5. 深度相机外参 = 彩色外参 · 出厂深度→彩色外参；仿真里用的是深度相机（可见性/盲区都按深度算）
#   6. 基座系核对：SDK 法兰位姿与 MJCF gripper_base 之间应是固定变换，算它在各样本上的离散度
# =============================================================================


def handeye_sessions() -> list[Path]:
    """All real (not merged / not sim) capture sessions, oldest first."""
    return sorted(p for p in CALIB_DATA_DIR.glob("*_handeye") if (p / "samples.jsonl").exists())


def merge_sessions(sessions: list[Path]) -> Path:
    """Write one merged session (group = source session index) so later steps can treat it like any session."""
    infos = [read_json(s / "camera_info.json") for s in sessions]
    boards = [read_json(s / "session.json")["board"] for s in sessions]
    serials = {i["device"].get("serial") for i in infos}
    if len(serials) != 1:
        raise SystemExit(f"这些会话来自不同相机 {serials}，不能合并")
    if any(b != boards[0] for b in boards) or any(i["color_intrinsics"] != infos[0]["color_intrinsics"] for i in infos):
        raise SystemExit("这些会话的标定板或彩色内参不一致，不能合并")
    out = sessions[-1].parent / f"{time.strftime('%Y%m%d_%H%M%S')}_handeye_merged"   # 放在源会话旁边
    out.mkdir(parents=True)
    write_json(out / "camera_info.json", infos[0])
    write_json(out / "session.json", dict(kind="eye_in_hand_merged", board=boards[0],
                                          sources=[str(s.resolve()) for s in sessions], created=time.strftime("%F %T")))
    n = 0
    for g, s in enumerate(sessions):
        for r in read_jsonl(s / "samples.jsonl"):
            r.update(source=s.name, source_id=r["id"], id=n, group=g,
                     image=str((s / r["image"]).resolve()),
                     depth=str((s / r["depth"]).resolve()) if r.get("depth") else None)
            append_jsonl(out / "samples.jsonl", r)
            n += 1
    print(f"合并 {len(sessions)} 个会话、{n} 张样本 → {out}")
    return out


def latest_session(kind="handeye") -> Path:
    cands = sorted(p for p in CALIB_DATA_DIR.glob(f"*_{kind}*") if (p / "samples.jsonl").exists())
    if not cands:
        raise SystemExit(f"{CALIB_DATA_DIR} 下没有 {kind} 采样数据，先跑 step1")
    return cands[-1]


def load_samples(session: Path, board: Chessboard, K, D, exclude=()):
    samples, dropped = [], []
    for r in read_jsonl(session / "samples.jsonl"):
        if r["id"] in exclude:
            dropped.append((r["id"], "手动排除"))
            continue
        corners = np.asarray(r["corners"], float)
        T_cb, rms = solve_pnp(corners, board, K, D)
        if rms > MAX_PNP_RMS_PX:
            dropped.append((r["id"], f"PnP {rms:.2f} px > {MAX_PNP_RMS_PX}"))
            continue
        s = Sample(np.asarray(r["T_base_gripper_base"]), T_cb, corners, rms, r["id"], group=int(r.get("group", 0)))
        s.extra["flange"] = None if r.get("T_base_flange_sdk") is None else np.asarray(r["T_base_flange_sdk"])
        samples.append(s)
    return samples, dropped


def board_intrinsics(session: Path, board: Chessboard, size) -> tuple[np.ndarray, np.ndarray, float]:
    obj, img = [], []
    for r in read_jsonl(session / "samples.jsonl"):
        obj.append(board.object_points().astype(np.float32))
        img.append(np.asarray(r["corners"], np.float32).reshape(-1, 1, 2))
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, tuple(size), None, None)
    return K, D.ravel(), float(rms)


def flange_check(samples) -> dict | None:
    offs = [inv(s.extra["flange"]) @ s.T_base_grip for s in samples if s.extra.get("flange") is not None]
    if len(offs) < 3:
        return None
    M = mean_pose(offs)
    dt = [np.linalg.norm(o[:3, 3] - M[:3, 3]) for o in offs]
    dr = [rot_deg(M[:3, :3].T @ o[:3, :3]) for o in offs]
    return dict(T_flange_gripper_base=M, trans_max_m=float(max(dt)), rot_max_deg=float(max(dr)),
                offset_m=float(np.linalg.norm(M[:3, 3])), offset_deg=rot_deg(M[:3, :3]))


def verdict(rep: dict, n: int) -> tuple[bool, list[str]]:
    msgs, ok = [], True
    c, mq, st = rep["consistency"], rep["motion"], rep["stability"]
    if n < MIN_SAMPLES:
        ok = False
        msgs.append(f"有效样本 {n} 张 < {MIN_SAMPLES}：再采几张")
    if mq["max_rel_rot_deg"] < MIN_ROTATION_SPAN_DEG:
        ok = False
        msgs.append(f"姿态旋转跨度 {mq['max_rel_rot_deg']:.0f}° < {MIN_ROTATION_SPAN_DEG:.0f}°：多做大角度倾斜")
    if min(mq["axis_diversity"]) < 0.25:
        msgs.append(f"旋转轴不够分散 {np.round(mq['axis_diversity'], 2).tolist()}：补几张绕相机光轴转动、以及前后/左右倾斜的姿态")
    if st["trans_max_m"] > MAX_X_INSTABILITY_M or st["rot_max_deg"] > MAX_X_INSTABILITY_DEG:
        ok = False
        msgs.append(f"相机外参不稳定：去掉一部分数据重算，X 最多变化 {st['trans_max_m'] * 1000:.1f} mm / {st['rot_max_deg']:.2f}°"
                    f"（要求 ≤ {MAX_X_INSTABILITY_M * 1000:.0f} mm / {MAX_X_INSTABILITY_DEG}°）：再采一个会话，用 --all-sessions 合并")
    if c["trans_max_m"] > GROSS_BOARD_SPREAD_M or c["rot_max_deg"] > GROSS_BOARD_SPREAD_DEG:
        ok = False
        msgs.append(f"有样本推算的板位姿偏差 {c['trans_max_m'] * 1000:.1f} mm / {c['rot_max_deg']:.2f}°，属于粗差："
                    f"会话中标定板被碰？相机支架松？拍照时机械臂在动？")
    elif c["trans_max_m"] > MAX_BOARD_SPREAD_M or c["rot_max_deg"] > MAX_BOARD_SPREAD_DEG:
        msgs.append(f"（提示）各样本推算的板位姿最大偏差 {c['trans_max_m'] * 1000:.1f} mm / {c['rot_max_deg']:.2f}°："
                    f"这是机械臂名义运动学与真实位姿的差（绝对精度），多采也不会变小；只要上面 X 稳定即可")
    if rep["method_spread_m"] > 0.003:
        msgs.append(f"不同求解方法相差 {rep['method_spread_m'] * 1000:.1f} mm：姿态多样性可能不足")
    return ok, msgs


def main(argv=None):
    ap = argparse.ArgumentParser(description="手眼标定第 2 步：求解 + 质量判定（+ 写入仿真配置）")
    ap.add_argument("--session", nargs="+", default=None, help="step1 的输出目录（默认最新）；给多个就合并")
    ap.add_argument("--all-sessions", action="store_true", help="合并 calib_data 下全部真机手眼会话（推荐）")
    ap.add_argument("--intrinsics", choices=["factory", "board"], default="factory",
                    help="彩色内参来源：出厂（默认）或用这批棋盘格图自标")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], help="手动排除的样本 id")
    ap.add_argument("--no-auto-reject", action="store_true")
    ap.add_argument("--apply", action="store_true", help="通过验收后写入 algorithms/config/dc1_site.yaml")
    ap.add_argument("--force", action="store_true", help="未通过验收也写入（不推荐）")
    ap.add_argument("--site-config", default=str(SITE_CONFIG), help="写入哪个配置文件（默认 dc1_site.yaml）")
    args = ap.parse_args(argv)

    if args.all_sessions:
        srcs = handeye_sessions()
        if not srcs:
            raise SystemExit("calib_data 下没有真机手眼会话")
        session = merge_sessions(srcs) if len(srcs) > 1 else srcs[0]
    elif args.session and len(args.session) > 1:
        session = merge_sessions([Path(s) for s in args.session])
    else:
        session = Path(args.session[0]) if args.session else latest_session()
    info = read_json(session / "camera_info.json")
    board = Chessboard(**read_json(session / "session.json")["board"])
    if info.get("color_intrinsics"):
        color = Intrinsics.from_dict(info["color_intrinsics"])
    else:                                            # 没有出厂彩色内参 → 只能自标
        prof = info["profiles"]["color"]
        color = Intrinsics(prof["width"], prof["height"], 1.0, 1.0, 0.0, 0.0)
        if args.intrinsics != "board":
            print("没有出厂彩色内参，自动改用棋盘格自标（--intrinsics board）")
            args.intrinsics = "board"
    K, D = color.K, color.D
    if args.intrinsics == "board":
        K, D, rms = board_intrinsics(session, board, (color.width, color.height))
        print(f"用棋盘格自标彩色内参：RMS {rms:.3f} px, fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    samples, dropped = load_samples(session, board, K, D, set(args.exclude))
    if len(samples) < 4:
        raise SystemExit(f"有效样本只有 {len(samples)} 张，无法求解。丢弃原因: {dropped}")

    rep = calibrate(samples, board, K, D, with_stability=False)
    while not args.no_auto_reject:
        r = np.asarray(rep["reprojection_px"])
        worst = int(np.argmax(r))
        if r[worst] <= max(3 * np.median(r), 2.0) or len(dropped) >= 0.2 * (len(samples) + len(dropped)):
            break
        dropped.append((samples[worst].sample_id, f"重投影离群 {r[worst]:.1f} px"))
        samples.pop(worst)
        rep = calibrate(samples, board, K, D, with_stability=False)
    print("计算稳定性（去掉一部分数据重算 X）…")
    rep["stability"] = stability(samples, rep["X"], rep["B_groups"], board, K, D)

    X_color = rep["X"]
    T_color_depth = np.asarray(info["T_color_depth"]) if info.get("T_color_depth") else None
    X_depth = X_color @ T_color_depth if T_color_depth is not None else None
    ok, msgs = verdict(rep, len(samples))
    fc = flange_check(samples)
    c = rep["consistency"]

    print(f"\n==== 手眼标定结果（{session.name}）====")
    print(f"有效样本 {len(samples)} 张（{rep['n_groups']} 个会话）；丢弃 {len(dropped)} 张 {dropped if dropped else ''}")
    st = rep["stability"]
    print(f"相机外参稳定性（{st['method']}）：X 最多变化 {st['trans_max_m'] * 1000:.2f} mm / {st['rot_max_deg']:.3f}°")
    print(f"板位姿一致性：平移 RMS {c['trans_rms_m'] * 1000:.2f} mm（最大 {c['trans_max_m'] * 1000:.2f}），"
          f"旋转 RMS {c['rot_rms_deg']:.3f}°（最大 {c['rot_max_deg']:.3f}）")
    print(f"整链重投影 RMS {rep['reprojection_rms_px']:.2f} px；旋转跨度 {rep['motion']['max_rel_rot_deg']:.0f}°；"
          f"方法间差异 {rep['method_spread_m'] * 1000:.2f} mm（闭式最优: {rep['best_closed_form']}）")
    print("彩色相机在 gripper_base 下（OpenCV 系）：t =", np.round(X_color[:3, 3], 4).tolist())
    if X_depth is not None:
        print("深度相机在 gripper_base 下（OpenCV 系）：t =", np.round(X_depth[:3, 3], 4).tolist())
    else:
        msgs.append("没有出厂深度→彩色外参，深度相机外参无法推出")
    if fc:
        print(f"基座系核对：SDK 法兰 → gripper_base 固定偏移 {fc['offset_m'] * 1000:.1f} mm / {fc['offset_deg']:.2f}°，"
              f"各样本离散 {fc['trans_max_m'] * 1000:.2f} mm / {fc['rot_max_deg']:.3f}°")
        if fc["trans_max_m"] > 0.002 or fc["rot_max_deg"] > 0.2:
            msgs.append("SDK 法兰位姿与 MJCF 正解之间不是固定变换：本工程运动学与真机可能不一致，先别用于训练，查原因")
    truth = info.get("sim_truth")
    if truth:
        err_t = np.linalg.norm(X_color[:3, 3] - np.asarray(truth["X_color"])[:3, 3])
        err_r = rot_deg(X_color[:3, :3].T @ np.asarray(truth["X_color"])[:3, :3])
        print(f"[仿真干跑] 与真值相差 {err_t * 1000:.2f} mm / {err_r:.3f}°")
    print("结论：" + ("✅ 通过" if ok else "❌ 未通过"))
    for m in msgs:
        print("  -", m)

    result = dict(session=str(session), board=board.to_dict(), intrinsics_source=args.intrinsics,
                  color_intrinsics=dict(width=color.width, height=color.height, fx=float(K[0, 0]), fy=float(K[1, 1]),
                                        cx=float(K[0, 2]), cy=float(K[1, 2]), dist=np.asarray(D).ravel().tolist()),
                  depth_intrinsics=info.get("depth_intrinsics"), T_color_depth=T_color_depth,
                  T_gripper_base_color_cv=X_color, T_gripper_base_depth_cv=X_depth, T_base_board=rep["B"],
                  T_base_board_groups={str(g): B for g, B in rep["B_groups"].items()}, stability=rep["stability"],
                  accepted=ok, messages=msgs, n_samples=len(samples), dropped=dropped,
                  consistency={k: v for k, v in c.items() if k not in ("B", "B_groups")}, reprojection_px=rep["reprojection_px"],
                  reprojection_rms_px=rep["reprojection_rms_px"], method_spread_m=rep["method_spread_m"],
                  motion=rep["motion"], flange_check=fc, sample_ids=[s.sample_id for s in samples])
    write_json(session / "result.json", result)
    print(f"结果已写入 {session / 'result.json'}")

    if args.apply:
        if not (ok or args.force):
            raise SystemExit("未通过验收，没有写入仿真配置（确认要写就加 --force）")
        if X_depth is None or not info.get("depth_intrinsics"):
            raise SystemExit("缺深度相机外参/内参，不能写入仿真配置")
        path = update_site_config({"camera": camera_overrides(X_depth, info["depth_intrinsics"])},
                                  dict(step="handeye", session=str(session), accepted=ok,
                                       n_samples=len(samples), n_sessions=rep["n_groups"],
                                       x_instability_mm=round(st["trans_max_m"] * 1000, 2),
                                       x_instability_deg=round(st["rot_max_deg"], 3),
                                       board_spread_mm=round(c["trans_max_m"] * 1000, 2),
                                       reprojection_rms_px=round(rep["reprojection_rms_px"], 3)),
                                  path=Path(args.site_config))
        print(f"已写入仿真配置 {path}（深度相机位姿与内参）")
        print(f"下一步：python -m nero_vs_rl.sim2real.calibration.step3_verify_in_sim --session {session}")
    return result


if __name__ == "__main__":
    main()
