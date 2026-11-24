"""Step 2 (offline stage) - replay a recording through perception -> observation -> policy, without moving the arm.

  python -m nero_vs_rl.sim2real.deploy.step2_replay --record <录制目录> --model <模型.zip> [--video]
  python -m nero_vs_rl.sim2real.deploy.step2_replay --latest --model <模型.zip> --video

输出（写进录制目录）：replay_report.json（逐帧 + 汇总），replay.mp4（深度图上叠加检测框、抓取点和策略想走的方向）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ... import REAL_DATA_DIR, SIM_OUTPUT_DIR, TRAIN_CONFIG
from ...envs.observation import GraspTracker, assemble_obs, gripper_points
from ...utils.config import load_config
from ...utils.geometry import make_T
from .common import JointSource, RobotTwin, load_record, measured_belt_z, read_depth, write_json
from .perception import BeltObjectDetector, observe, size_in_training_range

# =============================================================================
# 【说明】离线回放检查什么（全部不需要机械臂动）：
#   感知   检出率；尺寸/类型是否稳定、是否在训练范围内；带面高度；估计速度的大小和方向（应沿 −Y、2～8 cm/s）
#   观测   每帧拼出的观测字典必须落在训练时的 observation_space 里；TCP 位姿离仿真初始位姿多远
#   策略   看得到物体时，策略指令的平移方向是否朝着最近的抓取（cos > 0.5 记为"朝向正确"）；
#          远离物体时不应该闭合夹爪
#   仿真录制（--sim 录的）还有真值：直接给出检测位置/尺寸误差
# 注意：机械臂没有按策略动，所以这是"开环"检查——每帧都从实测关节出发问策略"你想怎么动"，
#   上一步动作 obs["action"] 用策略上一帧的输出。闭环效果要到低速空跑阶段才能看。
# =============================================================================

TOWARD_COS = 0.5
# 当前准备上机的模型；换模型用 --model
DEFAULT_MODEL = SIM_OUTPUT_DIR / "sac_dc1_v5" / "best" / "best_model.zip"   # 2026-11-24 闭环干跑 300 回合：v5 现场带速 89.5%，v6 77.6%，真机先用 v5


def latest_record(root: Path = REAL_DATA_DIR) -> Path:
    recs = sorted(p for p in Path(root).glob("*record*") if (p / "frames.jsonl").exists())
    if not recs:
        raise SystemExit(f"{root} 下没有录制，先运行 step1_record")
    return recs[-1]


def sim_visible(truth: dict, twin: RobotTwin, cfg, cam_pos, cam_mat) -> bool:
    """Would the simulation have shown grasps for the true object in this frame? (FOV + depth range rule)"""
    from ...grasp_detection import analytic_grasps as gr
    from ...sensors import camera as cam
    h = np.asarray(truth["half"])
    gs = (gr.cylinder_grasps(h[0], h[2], twin.max_width, cfg.grasp_obs.max_tilt_deg) if truth["obj_type"] == "cylinder"
          else gr.box_grasps(h, twin.max_width, cfg.grasp_obs.max_tilt_deg))
    T = np.asarray(truth["T_wo"])
    uv, d = cam.project((T[:3, :3] @ gs.vis_obj.T).T + T[:3, 3], cam_pos, cam_mat, cfg.camera)
    c, m = cfg.camera, cfg.camera.image_margin_px
    return bool(np.any((d >= c.min_depth_m) & (d <= c.max_depth_m) & (uv[:, 0] >= m) & (uv[:, 0] <= c.width - m)
                       & (uv[:, 1] >= m) & (uv[:, 1] <= c.height - m)))


def draw(depth, det, Gw_obs, visible, K, T_cam_base, text_lines, dp_base=None, p_tcp=None):
    v = np.clip(depth / 0.8, 0, 1)
    img = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth <= 0] = 0
    rv, tv = cv2.Rodrigues(T_cam_base[:3, :3])[0], T_cam_base[:3, 3]

    def proj(P):
        P = np.asarray(P, float).reshape(-1, 3)
        z = P @ T_cam_base[2, :3] + T_cam_base[2, 3]
        uv = cv2.projectPoints(P, rv, tv, K, None)[0].reshape(-1, 2)
        return uv, z > 0.01

    if det is not None:
        T = det.T_wo
        hx, hy, hz = det.half
        corners = np.array([[sx * hx, sy * hy, hz] for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
        uv, ok = proj(corners @ T[:3, :3].T + T[:3, 3])
        if ok.all():
            cv2.polylines(img, [uv.astype(np.int32)], True, (255, 255, 255) if not det.truncated else (128, 128, 128), 2)
    if len(Gw_obs):
        uv, ok = proj(Gw_obs[:, :3, 3])
        for (u, vv), o in zip(uv, ok):
            if o:
                cv2.circle(img, (int(u), int(vv)), 3, (0, 255, 0) if visible else (0, 255, 255), -1)
    if dp_base is not None and p_tcp is not None:
        uv, ok = proj(np.stack([p_tcp, p_tcp + dp_base * 5]))          # 放大 5 倍画方向（品红箭头）
        if ok.all():
            cv2.arrowedLine(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), (255, 0, 255), 2, tipLength=0.3)
    for k, s in enumerate(text_lines):
        cv2.putText(img, s, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def replay(record: Path, model_path: str | None, config: str, video: bool = False, seed: int = 0,
           latency: float | None = None) -> dict:
    meta, frames = load_record(record)
    cfg, _ = load_config(config)
    # 【说明】相机延迟只属于真机：仿真录制的深度图就是按录下的关节角渲染的（没有延迟），
    #   而且每个回合时间戳从 0 重来，按时间插值关节会取错。所以配置里的 latency_s 只用于真机录制；
    #   --latency 显式给出时照用（对比实验用）
    if latency is None:
        latency = cfg.camera.latency_s if meta.get("source") == "real" else 0.0
    joints = JointSource(record, frames)
    twin = RobotTwin(cfg)
    di = meta["camera"]["depth_intrinsics"]
    K = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
    D = np.asarray(di.get("dist") or [0.0] * 5, float)
    detector = BeltObjectDetector(cfg, K, D, twin.max_width)
    tracker = GraspTracker(cfg.grasp_obs, twin.control_dt)
    rng = np.random.default_rng(seed)
    model = None
    if model_path:
        from stable_baselines3 import SAC
        model = SAC.load(model_path, device="cpu")
    space = twin.env.observation_space
    rows, video_frames, ep_prev = [], [], None
    prev_action, t0 = np.zeros(7, dtype=np.float32), frames[0]["stamp"] if frames else 0.0
    p_home, R_home = twin.home_tcp()
    for rec in frames:
        if rec.get("episode", 0) != ep_prev:                        # 新回合（仿真录制里换物体）
            ep_prev = rec.get("episode", 0)
            detector.reset()
            tracker.reset()
            prev_action = np.zeros(7, dtype=np.float32)
        t = rec["stamp"] - t0
        q = joints.at(rec["stamp"] - latency) if latency else np.asarray(rec["joints"], float)
        twin.set_joints(q)
        p, R = twin.tcp_pose()
        T_bc = twin.T_base_cam_cv()
        cp, cm = twin.cam_pose_mj()
        depth = read_depth(record, rec)
        det, Gw, sc, why = observe(detector, depth, T_bc, make_T(R, p), cp, cm, cfg)
        if detector.new_object:                                     # 换了物体：记忆和测速窗口重新开始
            tracker.reset()
        Gw_obs, sc_obs, visible = tracker.update(t, Gw, sc, det.center if Gw is not None else None)
        pts = gripper_points(cfg.grasp_obs, twin.K, rng, twin.fixed_pts)
        obs, grasps_ee, valid = assemble_obs(p, R, Gw_obs, sc_obs, visible, tracker.age_norm, 1.0,
                                             prev_action, pts, twin.N)
        row = dict(i=rec["i"], t=round(t, 3), episode=ep_prev, detected=det is not None, visible=bool(visible),
                   why=why, mem_age=int(min(tracker.mem_age, 999)), n_grasps=int(valid.sum()),
                   obs_in_space=bool(space.contains(obs)), tcp=p.tolist(),
                   vel_est=tracker.est_obj_vel.tolist())
        if det is not None:
            row.update(object_id=detector.object_id, obj_type=det.obj_type, center=det.center.tolist(), half=det.half.tolist(),
                       yaw=det.yaw, truncated=det.truncated,
                       belt_z=float(det.center[2] - det.half[2]))          # 物体正下方的带面高度
        if valid.any():
            row["nearest_grasp_ee"] = grasps_ee[0, :3, 3].tolist()
        cmd = None
        if model is not None:
            # 连续查询：obs["action"] = 策略上一帧输出（与闭环时一样），但机械臂没动，只作参考
            a = model.predict(obs, deterministic=True)[0].astype(np.float32)
            row.update(action=a.tolist(), grip_cmd=float(a[6]))
            prev_action = a
            cmd = twin.command(q, a)
            if visible:
                # 首帧式查询："如果回合从这一帧开始，策略往哪走"——上一步动作清零，与训练时回合开头一致
                a0 = model.predict(dict(obs, action=np.zeros(7, dtype=np.float32)), deterministic=True)[0]
                cmd = twin.command(q, a0)
                row.update(action_fresh=a0.tolist(), dp_ee=cmd["dp_ee"].tolist(), grip_fresh=float(a0[6]))
                g = grasps_ee[0, :3, 3]
                if np.linalg.norm(g) > 0.03 and np.linalg.norm(cmd["dp_ee"]) > 1e-4:
                    row["toward_cos"] = float(g @ cmd["dp_ee"] / (np.linalg.norm(g) * np.linalg.norm(cmd["dp_ee"])))
        truth = rec.get("truth")
        if truth is not None:
            row["sim_visible"] = sim_visible(truth, twin, cfg, cp, cm)
        if truth is not None and det is not None:
            Tt = np.asarray(truth["T_wo"])
            row["truth"] = dict(obj_type=truth["obj_type"], center_err_mm=float(np.linalg.norm(det.center - Tt[:3, 3]) * 1000),
                                size_err_mm=float(np.max(np.abs(np.sort(det.half[:2]) - np.sort(np.asarray(truth["half"])[:2]))) * 1000),
                                height_err_mm=float(abs(det.half[2] - truth["half"][2]) * 1000),
                                type_ok=det.obj_type == truth["obj_type"])
        rows.append(row)
        if video:
            lines = [f"t={t:5.1f}s  {'VISIBLE' if visible else ('memory age %d' % tracker.mem_age if len(Gw_obs) else 'no target')}",
                     (f"{det.obj_type} {2 * det.half[0] * 100:.1f}x{2 * det.half[1] * 100:.1f}x{2 * det.half[2] * 100:.1f}cm"
                      + (" (cut)" if det.truncated else "")) if det is not None else "no detection",
                     f"v=({tracker.est_obj_vel[0] * 100:+.1f},{tracker.est_obj_vel[1] * 100:+.1f}) cm/s"]
            if "grip_fresh" in row:
                lines.append(f"policy: grip {'CLOSE' if row['grip_fresh'] < 0 else 'open'}  cos={row.get('toward_cos', float('nan')):+.2f}")
            video_frames.append(draw(depth, det, Gw_obs, visible, K, np.linalg.inv(T_bc), lines,
                                     None if cmd is None else cmd["dp_base"],
                                     np.array([p[0], p[1], float(detector.plane_z(p[:2]))])))   # 箭头从 TCP 正下方的带面画起
    twin.close()
    summary = summarize(rows, cfg, meta, p_home, R_home, frames,
                        measured_belt_z(config) if meta.get("source") == "real" else None)   # 仿真带面就是配置值
    summary["camera_latency_s"] = latency
    write_json(record / "replay_report.json", dict(summary=summary, frames=rows, model=model_path, config=config))
    if video and video_frames:
        import imageio.v2 as imageio
        imageio.mimsave(record / "replay.mp4", video_frames, fps=meta.get("hz", 10))
        summary["video"] = str(record / "replay.mp4")
    return summary


def summarize(rows, cfg, meta, p_home, R_home, frames, belt_z_ref: float | None = None) -> dict:
    n = len(rows)
    det = [r for r in rows if r["detected"]]
    vis = [r for r in rows if r["visible"]]
    s = dict(source=meta.get("source"), frames=n, detected_frac=len(det) / max(n, 1), visible_frac=len(vis) / max(n, 1),
             obs_all_in_space=all(r["obs_in_space"] for r in rows), warnings=[])
    if frames:
        twin_q = np.asarray(frames[0]["joints"])
        s["first_joints"] = twin_q.tolist()
    p0 = np.asarray(rows[0]["tcp"]) if rows else None
    if p0 is not None:
        s["tcp_offset_from_sim_home_mm"] = float(np.linalg.norm(p0 - p_home) * 1000)
        if s["tcp_offset_from_sim_home_mm"] > 1000 * cfg.init_pos_noise_m + 20:
            s["warnings"].append(f"起始 TCP 离仿真初始位姿 {s['tcp_offset_from_sim_home_mm']:.0f} mm，"
                                 f"超出训练时的随机范围（±{cfg.init_pos_noise_m * 100:.0f} cm）：先把机械臂摆回等待位")
    good = [r for r in det if not r["truncated"]]
    objs = []
    for oid in sorted({r["object_id"] for r in good}):
        seg = [r for r in good if r["object_id"] == oid]
        if len(seg) < 3:
            continue
        types = [r["obj_type"] for r in seg]
        halves = np.array([r["half"] for r in seg])
        o = dict(object_id=oid, frames=len(seg), t=[seg[0]["t"], seg[-1]["t"]],
                 obj_type=max(set(types), key=types.count), type_votes={k: types.count(k) for k in set(types)},
                 size_cm=(2 * np.median(halves, axis=0) * 100).round(2).tolist(),
                 size_std_cm=(2 * halves.std(axis=0) * 100).round(2).tolist(),
                 belt_plane_z=float(np.median([r["belt_z"] for r in seg])))
        lat = np.array([r["center"][0] for r in seg]) - cfg.belt_center_x
        o["lateral_offset_cm"] = [round(float(lat.min() * 100), 1), round(float(lat.max() * 100), 1)]
        vis_seg = [r for r in rows if r["visible"] and r.get("object_id") == oid][3:]
        if vis_seg:
            v = np.median(np.array([r["vel_est"] for r in vis_seg]), axis=0)
            o["vel_est_cm_s"] = (v * 100).round(2).tolist()
            o["speed_cm_s"] = round(float(np.linalg.norm(v[:2])) * 100, 2)
            o["direction_dev_deg"] = round(float(np.degrees(np.arccos(np.clip(
                -v[1] * np.sign(-cfg.belt_direction_sign) / max(np.linalg.norm(v[:2]), 1e-9), -1, 1)))), 1)
        objs.append(o)
        from .perception import Detection
        d = Detection(o["obj_type"], np.asarray(seg[-1]["center"]), seg[-1]["yaw"], np.median(halves, axis=0),
                      0, False, 1.0, np.zeros(3))
        tag = f"物体 {oid}（{o['obj_type']}）："
        s["warnings"] += [tag + w for w in size_in_training_range(d, cfg)]
        lat_max = cfg.belt_half_width if cfg.trajectory.lateral_full_belt else max(abs(x) for x in cfg.trajectory.lateral)
        if np.abs(lat).max() > lat_max + 0.02:
            s["warnings"].append(tag + f"偏离带中心线 {np.abs(lat).max() * 100:.1f} cm，超出训练范围 ±{lat_max * 100:.0f} cm")
        if "speed_cm_s" in o:
            lo, hi = cfg.trajectory.speed
            if not (lo * 100 * 0.8 <= o["speed_cm_s"] <= hi * 100 * 1.2):
                s["warnings"].append(tag + f"估计带速 {o['speed_cm_s']:.1f} cm/s 不在训练范围 {lo * 100:.0f}～{hi * 100:.0f} cm/s")
            if o["speed_cm_s"] > 1.0 and o["direction_dev_deg"] > 15:
                s["warnings"].append(tag + f"估计运动方向偏离 −Y {o['direction_dev_deg']:.0f}°：检查手眼 / 带方向")
        # 带面高度和标定 step5 用同一个相机测得的值比：差 5 mm 以上，多半是相机被碰动了
        if belt_z_ref is not None and abs(o["belt_plane_z"] - belt_z_ref) > 0.005:
            s["warnings"].append(tag + f"物体下方带面 z={o['belt_plane_z']:.3f}，和标定时测得的 {belt_z_ref:.3f} 差 "
                                 f"{abs(o['belt_plane_z'] - belt_z_ref) * 1000:.0f} mm：相机可能被碰动，检查后重做标定 step1～step3")
        if o["size_std_cm"][0] > 0.5 or o["size_std_cm"][1] > 0.5:
            s["warnings"].append(tag + f"逐帧尺寸波动 {o['size_std_cm']} cm 偏大：深度噪声或物体表面反光")
    s["objects"] = objs
    fresh = [r for r in rows if "action_fresh" in r]
    if fresh:
        cos = np.array([r["toward_cos"] for r in fresh if "toward_cos" in r])
        s["policy_toward_frac"] = float((cos > TOWARD_COS).mean()) if len(cos) else None
        s["policy_toward_frames"] = int(len(cos))
        far = [r for r in fresh if np.linalg.norm(r["nearest_grasp_ee"]) > 0.10]
        s["close_cmd_when_far_frac"] = float(np.mean([r["grip_fresh"] < 0 for r in far])) if far else None
        if s["close_cmd_when_far_frac"] and s["close_cmd_when_far_frac"] > 0.2:
            s["warnings"].append("看得到物体、离物体 >10 cm 时策略经常要闭合夹爪：观测可能和训练不一致")
        if s["policy_toward_frac"] is not None and s["policy_toward_frac"] < 0.6:
            s["warnings"].append(f"看得到物体时只有 {s['policy_toward_frac'] * 100:.0f}% 的帧朝着抓取移动")
    blind = [r for r in rows if "action" in r and r["n_grasps"] == 0]
    if blind:
        # 训练里"没有任何抓取信息"几乎只发生在已经抓住、物体进了盲区的时候 → 策略学成"闭合 + 抬起"
        s["no_target_close_lift_frac"] = float(np.mean([r["action"][6] < 0 and r["action"][2] < 0 for r in blind]))
        if s["no_target_close_lift_frac"] > 0.5:
            s["warnings"].append(f"没有目标时策略 {s['no_target_close_lift_frac'] * 100:.0f}% 的帧要求闭合夹爪并后退："
                                 "上真机时要等第一次完整检出物体再启动策略")
    sv = [r for r in rows if r.get("sim_visible")]
    if sv:
        s["sim_visible_but_unusable_frac"] = float(np.mean([not r["visible"] for r in sv]))
    tr = [r["truth"] for r in rows if "truth" in r and r["visible"]]
    if tr:
        s["truth_center_err_mm_median"] = float(np.median([x["center_err_mm"] for x in tr]))
        s["truth_size_err_mm_median"] = float(np.median([x["size_err_mm"] for x in tr]))
        s["truth_type_acc"] = float(np.mean([x["type_ok"] for x in tr]))
    if not s["obs_all_in_space"]:
        s["warnings"].append("有观测超出 observation_space：检查数据")
    return s


def print_summary(s: dict):
    print(f"\n==== 离线回放（{s['source']}，{s['frames']} 帧）====")
    print(f"检出 {s['detected_frac'] * 100:.0f}% 帧，可用于观测（未截断、有可见抓取）{s['visible_frac'] * 100:.0f}% 帧")
    for o in s.get("objects", []):
        v = f"，速度 {o['vel_est_cm_s']} cm/s（偏离 −Y {o['direction_dev_deg']}°）" if "vel_est_cm_s" in o else ""
        print(f"物体 {o['object_id']}：{o['obj_type']} {o['size_cm']} cm（逐帧标准差 {o['size_std_cm']}），{o['frames']} 帧，"
              f"带面 z≈{o['belt_plane_z']:.3f} m，偏离中心线 {o['lateral_offset_cm']} cm{v}")
    if "tcp_offset_from_sim_home_mm" in s:
        print(f"起始 TCP 离仿真初始位姿 {s['tcp_offset_from_sim_home_mm']:.0f} mm")
    if s.get("policy_toward_frac") is not None:
        print(f"策略：看得到物体的 {s['policy_toward_frames']} 帧里 {s['policy_toward_frac'] * 100:.0f}% 朝向抓取；"
              + (f"远处要闭合夹爪 {s['close_cmd_when_far_frac'] * 100:.0f}%" if s["close_cmd_when_far_frac"] is not None else "")
              + "（首帧式查询）")
    if "no_target_close_lift_frac" in s:
        print(f"没有目标的帧里策略要求闭合并后退的比例 {s['no_target_close_lift_frac'] * 100:.0f}%")
    if "sim_visible_but_unusable_frac" in s:
        print(f"（仿真真值）仿真里能看到抓取、但感知不可用的帧占 {s['sim_visible_but_unusable_frac'] * 100:.0f}%"
              "（多为物体刚进画面、被画面边缘截断）")
    if "truth_center_err_mm_median" in s:
        print(f"（仿真真值）位置误差中位数 {s['truth_center_err_mm_median']:.1f} mm，尺寸误差 {s['truth_size_err_mm_median']:.1f} mm，"
              f"类型正确率 {s['truth_type_acc'] * 100:.0f}%")
    print("观测全部在训练空间内" if s["obs_all_in_space"] else "⚠ 有观测超出训练空间")
    for w in s["warnings"]:
        print("⚠", w)
    if not s["warnings"]:
        print("没有发现问题")
    if "video" in s:
        print("视频:", s["video"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="离线回放：录制数据 → 感知 → 观测 → 策略（机械臂不动）")
    ap.add_argument("--record", nargs="+", default=None, help="step1 的录制目录（可以给多个）")
    ap.add_argument("--latest", action="store_true", help=f"用 {REAL_DATA_DIR} 下最新的录制")
    ap.add_argument("--all", action="store_true", help="回放所有还没回放过的真机录制（没有 replay_report.json 的）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else None,
                    help=f"SAC 模型 .zip（默认 {DEFAULT_MODEL}）；给 none 则只检查感知和观测")
    ap.add_argument("--config", default=str(TRAIN_CONFIG))
    ap.add_argument("--video", action="store_true", help="输出 replay.mp4")
    ap.add_argument("--latency", type=float, default=None, help="相机延迟（秒），默认用配置里的 camera.latency_s")
    args = ap.parse_args(argv)
    if args.model and args.model.lower() == "none":
        args.model = None
    if args.record:
        records = [Path(r) for r in args.record]
    elif args.latest:
        records = [latest_record()]
    elif args.all:
        records = [p for p in sorted(Path(REAL_DATA_DIR).glob("*_record"))
                   if (p / "frames.jsonl").exists() and not (p / "replay_report.json").exists()]
        if not records:
            raise SystemExit(f"{REAL_DATA_DIR} 下没有未回放的真机录制（已回放的用 --record 指定目录重跑）")
    else:
        ap.error("用 --latest（最新一段）、--all（所有没回放过的）或 --record 目录")
    print(f"配置: {args.config}\n模型: {args.model}")
    out = []
    for record in records:
        meta = load_record(record)[0]
        print(f"\n录制: {record}\n备注: {meta.get('note', '')}")
        s = replay(record, args.model, args.config, video=args.video, latency=args.latency)
        print_summary(s)
        print(f"逐帧结果: {record / 'replay_report.json'}")
        out.append(s)
    return out[0] if len(out) == 1 else out


if __name__ == "__main__":
    main()
