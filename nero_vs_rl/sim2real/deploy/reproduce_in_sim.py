"""Reproduce a real closed-loop grasp episode in the MuJoCo cell: same start joints, same static object, same profile.

  python -m nero_vs_rl.sim2real.deploy.reproduce_in_sim --record /home/yunfeiwu/adddisk1/MY_RL_test/real/20261124_032310_closed_loop_grasp
  python -m nero_vs_rl.sim2real.deploy.reproduce_in_sim --record /home/yunfeiwu/adddisk1/MY_RL_test/real/20261124_032310_closed_loop_grasp --cam-latency-ms 80 --latency-jitter-ms 10

用来回答"真机失败的这一回合，放到仿真里会不会也失败"：仿真也失败 → 策略/任务的问题；仿真成功 → 真机和仿真还有差别，
再逐项打开相机效应（延迟、帧时刻抖动、深度噪声、边缘缺失、盲区）找出是哪一项。只支持静止物体（带不开的回合）。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np

from ...utils.config import load_config
from ...utils.geometry import make_T
from ..hardware.sim_workcell import CameraEffects, SimWorkcell
from .common import JointSource, RobotTwin, load_record, read_depth
from .lab import safe_belt_z
from .perception import BeltObjectDetector, observe
from .runtime.safety import SafetyProfile
from .runtime.session import LoopSession, load_policy

# =============================================================================
# 【说明】从真机闭环日志里取出这一回合的"初始条件"：
#   起始关节   = 该回合第一拍的实测关节
#   物体       = 开头几帧完整检出的中位数：类型、中心 (x, y)、尺寸；朝向不在逐拍日志里，用同一份感知代码
#                在开头的深度帧上重新检测一次（同 step2 回放，按配置延迟补偿关节）
#   带面高度   = 物体中心 z − 半高（真机带面，仿真世界按它放）
#   档位、配置 = meta.json 里记录的（安全带面同真机：max(配置, 标定实测)）
#   然后在虚拟实验室里跑 N 次（只有策略的抓取点采样和相机效应的随机数不同），统计成功率和第一次闭合的位置。
# =============================================================================
def initial_conditions(record: Path, episode: int, cfg) -> dict:
    rows = [json.loads(l) for l in (record / "ticks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["ep"] == episode]
    usable = [r for r in rows if r.get("usable") and r.get("center") is not None][:5]
    if not usable:
        raise SystemExit("这一回合没有完整检出物体，无法复现")
    center = np.median([r["center"] for r in usable], axis=0)
    half = np.median([r["half"] for r in usable], axis=0)
    types = [r["obj_type"] for r in usable]
    ic = dict(q0=np.asarray(rows[0]["q"], float), obj_type=max(set(types), key=types.count), center=center, half=half,
              belt_z=float(center[2] - half[2]), yaw=None, real_rows=rows)
    meta, frames = load_record(record)
    joints, twin = JointSource(record, frames), RobotTwin(cfg)
    di = meta["camera"]["depth_intrinsics"]
    K = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
    det_ = BeltObjectDetector(cfg, K, np.asarray(di.get("dist") or [0.0] * 5, float), twin.max_width)
    yaws = []
    for rec in [f for f in frames if f.get("episode", 0) == episode][:12]:
        twin.set_joints(joints.at(rec["stamp"] - cfg.camera.latency_s))
        p, R = twin.tcp_pose()
        cp, cm = twin.cam_pose_mj()
        det, Gw, _, _ = observe(det_, read_depth(record, rec), twin.T_base_cam_cv(), make_T(R, p), cp, cm, cfg)
        if det is not None and Gw is not None:
            yaws.append(det.yaw)
            ic["half"] = np.asarray(det.half, float)                    # 尺寸和朝向用同一帧，轴对应得上
    if yaws:
        period = np.pi if ic["obj_type"] == "box" else 2 * np.pi        # 长方体朝向差 180° 是同一个
        a = np.angle(np.mean(np.exp(1j * np.array(yaws) * 2 * np.pi / period))) * period / (2 * np.pi)
        ic["yaw"] = float(a)
    return ic


def world_config(cfg, ic: dict, render_min_depth: float | None):
    h, c = ic["half"], ic["center"]
    tr = dataclasses.replace(cfg.trajectory, mode="static", lateral_full_belt=False,
                             lateral=[float(c[0] - cfg.belt_center_x)] * 2, start_along=[float(c[1])] * 2)
    if ic["obj_type"] == "box":
        ob = dataclasses.replace(cfg.objects, library=None, types=["box"], box_half_x=[float(h[0])] * 2, box_half_y=[float(h[1])] * 2,
                                 box_half_z=[float(h[2])] * 2)
    else:
        r = float((h[0] + h[1]) / 2)
        ob = dataclasses.replace(cfg.objects, library=None, types=["cylinder"], cyl_radius=[r] * 2, cyl_half_height=[float(h[2])] * 2)
    cam = cfg.camera if render_min_depth is None else dataclasses.replace(cfg.camera, min_depth_m=render_min_depth)
    return dataclasses.replace(cfg, trajectory=tr, objects=ob, camera=cam, belt_surface_z=ic["belt_z"],
                               init_pos_noise_m=0.0, init_yaw_noise_rad=0.0)


# 【说明】接触相关的敏感度实验：真机手指是海绵垫、物体是轻塑料盒，摩擦和质量都可能和仿真差很多；
#   手指实际长度（指尖离法兰的距离）也没用尺子核对过。这些只在复现时改，不动模型文件。
def physics_overrides(cell: SimWorkcell, args):
    m, env, ag = cell.model, cell.env, cell.agent
    if not hasattr(cell, "_orig_physics"):
        cell._orig_physics = (m.geom_friction.copy(), m.geom_pos.copy())
    m.geom_friction[:], m.geom_pos[:] = cell._orig_physics[0], cell._orig_physics[1]
    if args.finger_friction is not None:
        for g in ag.finger_geoms:
            m.geom_friction[g][0] = args.finger_friction
        if args.obj_friction is None:                        # 接触取两者较大值：物体不跟着降，手指就白改了
            o = env.obj["geom"]
            m.geom_friction[o][0] = min(m.geom_friction[o][0], args.finger_friction)
    if args.obj_friction is not None:
        m.geom_friction[env.obj["geom"]][0] = args.obj_friction
    if args.obj_mass is not None:
        b = env.obj["body"]
        m.body_inertia[b] *= args.obj_mass / m.body_mass[b]
        m.body_mass[b] = args.obj_mass
    if args.finger_shift_mm:
        mujoco.mj_forward(m, cell.data)
        approach = cell.data.site_xmat[m.site("ee_site").id].reshape(3, 3)[:, 2]     # TCP z = 指尖朝外的接近方向
        for g in ag.finger_geoms:
            Rb = cell.data.xmat[m.geom_bodyid[g]].reshape(3, 3)
            m.geom_pos[g] -= Rb.T @ approach * args.finger_shift_mm / 1000


def _jsonable(x):
    return x.tolist() if isinstance(x, np.ndarray) else float(x) if isinstance(x, np.generic) else str(x)


def first_close(rows: list[dict], obj_xyz) -> dict | None:
    """TCP relative to the object (cm) at the first tick the policy asks to close, and the gripper width 0.2 s later."""
    servo = [r for r in rows if r.get("action") is not None and r.get("tcp") is not None]
    for k, r in enumerate(servo):
        if r["action"][6] < 0:
            later = servo[min(k + 2, len(servo) - 1)]
            rel = (np.asarray(r["tcp"]) - np.asarray(obj_xyz)) * 100
            return dict(t=r["t"], rel_cm=np.round(rel, 1).tolist(), width_after_cm=round(later["width"] * 100, 1))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="在虚拟实验室里复现一次真机闭环抓取（静止物体）")
    ap.add_argument("--record", required=True, help="真机闭环日志目录（*_closed_loop_grasp）")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=16, help="仿真里重复几次")
    ap.add_argument("--model", default=None, help="默认用日志里记录的模型")
    ap.add_argument("--random-yaw", action="store_true", help="物体朝向每次随机（默认用重新检测出的真机朝向）")
    ap.add_argument("--cam-latency-ms", type=float, default=0.0)
    ap.add_argument("--latency-jitter-ms", type=float, default=0.0)
    ap.add_argument("--depth-noise-mm", type=float, default=0.0)
    ap.add_argument("--edge-dropout", type=float, default=0.0)
    ap.add_argument("--render-min-depth", type=float, default=None, help="仿真相机的真实盲区（感知仍按配置）")
    ap.add_argument("--finger-friction", type=float, default=None, help="手指滑动摩擦系数；没给 --obj-friction 时物体也降到不高于它（接触取较大值）")
    ap.add_argument("--obj-friction", type=float, default=None, help="物体滑动摩擦系数（模型 1.0）")
    ap.add_argument("--obj-mass", type=float, default=None, help="物体质量 kg（默认按训练范围随机）")
    ap.add_argument("--finger-shift-mm", type=float, default=0.0, help="手指碰撞体沿接近方向往回缩（正 = 手指变短）")
    ap.add_argument("--compute-delay-ms", type=float, default=0.0,
                    help="每拍从读关节/取帧到发出指令的耗时（真机中位 26～30 ms）；仿真时钟默认在计算时停住")
    ap.add_argument("--dump", default=None, help="把每次复现的逐拍日志（同真机 ticks.jsonl 字段 + 物体真值）写到这个 .jsonl")
    args = ap.parse_args(argv)
    record = Path(args.record)
    meta = json.loads((record / "meta.json").read_text(encoding="utf-8"))
    cfg, _ = load_config(meta["config"])
    ic = initial_conditions(record, args.episode, cfg)
    profile = SafetyProfile(**meta["profile"])
    real_close = first_close(ic["real_rows"], ic["center"])
    print(f"真机 {record.name} 回合 {args.episode}：{ic['obj_type']} 尺寸 {np.round(ic['half'] * 200, 1).tolist()} cm，"
          f"中心 {np.round(ic['center'], 3).tolist()}，朝向 {'?' if ic['yaw'] is None else round(np.degrees(ic['yaw']))}°，"
          f"带面 {ic['belt_z']:.3f}；第一次闭合 {real_close}")
    fx = CameraEffects(args.cam_latency_ms / 1000, args.latency_jitter_ms / 1000, args.depth_noise_mm / 1000,
                       args.edge_dropout, seed=args.episode)
    cell = SimWorkcell(world_config(cfg, ic, args.render_min_depth), camera_effects=fx)
    policy = load_policy(args.model or meta["model"])
    if args.compute_delay_ms > 0:                                   # 【说明】真机算一拍要 ~28 ms，这段时间机械臂照常在动
        raw_policy = policy

        def policy(obs):
            a = raw_policy(obs)
            cell.clock.sleep_until(cell.clock.now() + args.compute_delay_ms / 1000)
            return a
    rng = np.random.default_rng(0)
    session, results = None, []
    dump = open(args.dump, "w", encoding="utf-8") if args.dump else None
    try:
        for i in range(args.episodes):
            cell.reset(seed=1000 + i)
            yaw = rng.uniform(-np.pi, np.pi) if args.random_yaw or ic["yaw"] is None else ic["yaw"]
            physics_overrides(cell, args)
            cell.place_static_object(float(ic["center"][0]), float(ic["center"][1]), yaw)
            cell.set_arm(ic["q0"])
            if session is None:
                session = LoopSession(cfg, profile, cell.arm, cell.camera, cell.clock,
                                      belt_z_safe=safe_belt_z(cfg, meta["config"]), latency_s=fx.latency_s)
            else:
                session.streamer.reset_to(cell.arm.joints(), cell.clock.now())
            run = dict(n=0, ok=False)

            def on_tick(ep, row, run=run):
                run["n"] = run["n"] + 1 if cell.truth()["lifted"] else 0
                run["ok"] |= run["n"] >= 2
                return False

            obj0 = cell.data.xpos[cell.env.obj["body"]].copy()
            res = session.episode(policy, on_tick=on_tick, episode_index=i, seed=1000 + i).run()
            fc = first_close(res.rows, obj0)
            first_ok = bool(run["ok"]) and fc is not None and fc["width_after_cm"] > 1.5
            results.append(dict(ok=run["ok"], first_ok=first_ok, close=fc, outcome=res.outcome))
            if dump is not None:
                for r in res.rows:
                    dump.write(json.dumps(dict(r, ep=i, obj=obj0), ensure_ascii=False, default=_jsonable) + "\n")
            print(f"[{i:02d}] 朝向 {np.degrees(yaw):5.0f}°  {'抓起' if run['ok'] else '失败'}  {res.outcome:8s} 第一次闭合 {fc}")
    finally:
        if session is not None:
            session.close()
        cell.close()
        if dump is not None:
            dump.close()
    n = len(results)
    print(f"\n仿真复现 {n} 次：抓起 {sum(r['ok'] for r in results)}，第一次闭合就夹住 {sum(r['first_ok'] for r in results)}")
    rel = np.array([r["close"]["rel_cm"] for r in results if r["close"]])
    if len(rel):
        print(f"第一次闭合时 TCP − 物体（cm）：中位 {np.round(np.median(rel, 0), 1).tolist()}，"
              f"范围 x [{rel[:, 0].min():.1f}, {rel[:, 0].max():.1f}] y [{rel[:, 1].min():.1f}, {rel[:, 1].max():.1f}] "
              f"z [{rel[:, 2].min():.1f}, {rel[:, 2].max():.1f}]；真机 {real_close['rel_cm'] if real_close else '-'}")
    return results


if __name__ == "__main__":
    main()
