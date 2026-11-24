"""Step 4 (closed loop, no hardware) - run the whole lab loop against the MuJoCo cell before going to the lab.

  python -m nero_vs_rl.sim2real.deploy.step4_closed_loop_sim --episodes 10                       # 打开 MuJoCo 图形界面实时看
  python -m nero_vs_rl.sim2real.deploy.step4_closed_loop_sim --episodes 100 --no-viewer           # 统计
  python -m nero_vs_rl.sim2real.deploy.step4_closed_loop_sim --profile grasp --episodes 50 --no-viewer   # 用真机抓取档位跑
  python -m nero_vs_rl.sim2real.deploy.step4_closed_loop_sim --profile grasp --episodes 100 --no-viewer --lab-belt   # 安全带面按真机算（−0.112）
  python -m nero_vs_rl.sim2real.deploy.step4_closed_loop_sim --profile grasp --episodes 100 --no-viewer --cam-latency-ms 80 --depth-noise-mm 3 --edge-dropout 0.5   # 加真机相机效应

输出：SIM_OUTPUT_DIR/closed_loop_sim/<时间>_<档位>/（ticks.jsonl、episodes.json、summary.json；--frames 时另存深度帧，可用 step2 回放）
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from collections import Counter

import numpy as np

from ... import SIM_OUTPUT_DIR, TRAIN_CONFIG
from ...utils.config import SITE_BELT_SPEED_MAX, load_config
from ..hardware.sim_workcell import CameraEffects, SimWorkcell
from .common import write_json
from .lab import safe_belt_z
from .runtime.run_log import RunLog
from .runtime.safety import PROFILES
from .runtime.session import LoopSession, load_policy
from .step2_replay import DEFAULT_MODEL

# =============================================================================
# 【说明】为什么在去实验室之前先跑这一步：
#   真机闭环用到的每一段代码——在线感知（从深度图检出物体）、等目标、策略、同一份 IK、指尖抬高修正、
#   安全检查、100 Hz 插值发送、抓起判断、日志——这里全部原样运行，只是机械臂和相机换成了
#   MuJoCo 虚拟实验室（hardware/sim_workcell.py）。
#   和 sac_eval 的区别：sac_eval 走 env.step()（感知是"真值 + 噪声"），这里感知来自渲染的深度图，
#   控制走 100 Hz 发送，判断抓起只用夹爪开口和 TCP 高度（真机也有的量）。
#   所以这里的成功率 ≈ "这套部署代码 + 这个模型"在理想相机下能做到多少；和 sac_eval 差很多就说明部署链路有问题。
#   仿真真值（物体是否真的被抓起）只用来核对：抓起判断（GraspMonitor）有没有误报 / 漏报。
# =============================================================================
BANDS = [(0.0, 0.08), (0.08, 0.14), (0.14, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 9.0)]


def run(args) -> dict:
    cfg, _ = load_config(args.config)
    profile = PROFILES[args.profile]
    if args.speed_scale is not None:
        profile = dataclasses.replace(profile, speed_scale=args.speed_scale)
    if args.finger_clearance_mm is not None:
        profile = dataclasses.replace(profile, finger_clearance_m=args.finger_clearance_mm / 1000)
    out = SIM_OUTPUT_DIR / "closed_loop_sim" / f"{time.strftime('%Y%m%d_%H%M%S')}_{profile.name}"
    print(f"配置: {args.config}\n模型: {args.model}\n档位: {profile.name}（{profile.title}）\n输出: {out}")
    fx = CameraEffects(latency_s=args.cam_latency_ms / 1000, latency_jitter_s=args.latency_jitter_ms / 1000,
                       depth_noise_m=args.depth_noise_mm / 1000,
                       edge_dropout=args.edge_dropout, seed=args.seed)
    if not fx.ideal:
        print(f"相机效应: 延迟 {args.cam_latency_ms:.0f} ± {args.latency_jitter_ms:.0f} ms，深度噪声 {args.depth_noise_mm:.1f} mm @0.3 m，边缘缺失 {args.edge_dropout:.2f}")
    cell = SimWorkcell(cfg, viewer=args.viewer, playback=args.playback, camera_effects=fx)
    policy = load_policy(args.model)
    log = RunLog(out, dict(source="sim", hz=cfg.control_freq, config=str(args.config), model=args.model,
                           profile=dataclasses.asdict(profile), camera=dict(depth_intrinsics=cell.camera.intrinsics)),
                 save_frames=args.frames)
    cell.reset(seed=args.seed)
    # 仿真带面就在 cfg.belt_surface_z；--lab-belt 时安全层改按真机的安全带面算（标定实测更高时，指尖会比训练多抬这么多）
    belt_z = safe_belt_z(cfg, args.config) if args.lab_belt else cfg.belt_surface_z
    print(f"安全带面: {belt_z:.3f} m（仿真带面 {cfg.belt_surface_z:.3f}）")
    session = LoopSession(cfg, profile, cell.arm, cell.camera, cell.clock, belt_z_safe=belt_z,
                          latency_s=(args.cam_latency_ms if args.latency_comp_ms is None else args.latency_comp_ms) / 1000)
    eps = []
    try:
        for i in range(args.episodes):
            if i:
                cell.reset(seed=args.seed + i)
                session.streamer.reset_to(cell.arm.joints(), cell.clock.now())     # 仿真换回合把机械臂瞬移到新起点
            belt = cell.belt_speed()
            truth = dict(run=0, success_step=None)

            def on_tick(ep, row, truth=truth):
                tr = cell.truth()
                truth["run"] = truth["run"] + 1 if tr["lifted"] else 0          # 同 sac_eval：连续 2 拍抬起才算成功
                if truth["run"] >= 2 and truth["success_step"] is None:
                    truth["success_step"] = row.get("step")
                if cell.viewer is not None:
                    cell.viewer.set_text(f"STEP4 closed loop {i + 1}/{args.episodes}  {tr['obj_type']}  belt {belt * 100:.0f} cm/s",
                                         f"\n\n{ep.phase.name}  step {ep.steps}" + ("  LIFTED" if tr["lifted"] else ""))
                    if not cell.viewer.is_running():
                        return True
                return False

            res = session.episode(policy, on_tick=on_tick, log=log, episode_index=i, seed=args.seed + i).run()
            tr = cell.truth()
            rows = res.rows
            err = [np.abs(np.asarray(r["q"]) - np.asarray(r["q_cmd"])).max() for r in rows
                   if r.get("q_cmd") is not None and r["phase"] in ("SERVO", "HOLD")]
            e = dict(episode=i, obj=tr["obj_type"], belt_speed=belt, outcome=res.outcome, reasons=res.reasons,
                     truth_success=truth["success_step"] is not None, truth_held_end=tr["grasped"],
                     t_start_servo=res.t_start_servo, t_lifted=res.t_lifted, policy_steps=res.policy_steps,
                     max_track_err=float(max(err)) if err else 0.0,
                     lift_fix_ticks=sum(any("抬高" in f for f in r.get("fixes", [])) for r in rows),
                     limit_fix_ticks=sum(any("软限位" in f for f in r.get("fixes", [])) for r in rows),
                     compute_ms=float(np.median([r["compute_ms"] for r in rows])))
            eps.append(e)
            log.episode(e)
            print(f"[{i:03d}] {e['obj']:8s} 带速 {belt * 100:4.1f} cm/s  {res.outcome:9s} 真值{'成功' if e['truth_success'] else '失败'}"
                  f"  启动 {res.t_start_servo if res.t_start_servo is not None else float('nan'):.1f}s  策略 {res.policy_steps} 步"
                  f"  跟随误差 {e['max_track_err']:.2f} rad" + (f"  {res.reasons[0]}" if res.reasons else ""))
            if cell.viewer is not None:
                cell.viewer.pause(1.0)
                if not cell.viewer.is_running():
                    break
    finally:
        session.close()
        log.close()
        cell.close()
    s = summarize(eps, profile)
    write_json(out / "summary.json", s)
    print_summary(s)
    return s


def summarize(eps: list[dict], profile) -> dict:
    n = max(len(eps), 1)
    lifted = [e["outcome"] == "lifted" for e in eps]
    truth = [e["truth_success"] for e in eps]
    err = np.array([e["max_track_err"] for e in eps]) if eps else np.zeros(1)
    s = dict(profile=profile.name, episodes=len(eps), truth_success_rate=sum(truth) / n, monitor_lifted_rate=sum(lifted) / n,
             monitor_false_lift=sum(a and not b for a, b in zip(lifted, truth)),
             monitor_missed_lift=sum(b and not a for a, b in zip(lifted, truth)),
             outcomes=dict(Counter(e["outcome"] for e in eps)),
             stop_reasons=dict(Counter(r.split("：")[0] for e in eps for r in e["reasons"])),
             track_err_rad=dict(p50=float(np.median(err)), p95=float(np.percentile(err, 95)), max=float(err.max())),
             compute_ms_median=float(np.median([e["compute_ms"] for e in eps])) if eps else 0.0,
             episodes_with_finger_lift=sum(e["lift_fix_ticks"] > 0 for e in eps),
             episodes_with_limit_clip=sum(e["limit_fix_ticks"] > 0 for e in eps), bands=[])
    site = [e for e in eps if e["belt_speed"] <= SITE_BELT_SPEED_MAX]
    s["site_success_rate"] = sum(e["truth_success"] for e in site) / len(site) if site else None
    s["site_episodes"] = len(site)
    for lo, hi in BANDS:
        b = [e for e in eps if lo <= e["belt_speed"] < hi]
        if b:
            s["bands"].append(dict(band=f"{lo * 100:.0f}~{hi * 100:.0f}" if hi < 9 else f"{lo * 100:.0f}~",
                                   n=len(b), truth_success=sum(e["truth_success"] for e in b) / len(b)))
    return s


def print_summary(s: dict):
    print(f"\n==== 闭环干跑（{s['profile']} 档，{s['episodes']} 回合）====")
    print(f"真值成功率 {s['truth_success_rate'] * 100:.1f}%；抓起判断（只用夹爪开口和 TCP 高度）判为抓起 {s['monitor_lifted_rate'] * 100:.1f}%"
          f"，误报 {s['monitor_false_lift']}、漏报 {s['monitor_missed_lift']}")
    for b in s["bands"]:
        print(f"  带速 {b['band']:>6s} cm/s：{b['truth_success'] * 100:5.1f}%（{b['n']} 回合）")
    if s["site_success_rate"] is not None:
        print(f"现场带速（≤ {SITE_BELT_SPEED_MAX * 100:.0f} cm/s）：{s['site_success_rate'] * 100:.1f}%（{s['site_episodes']} 回合）")
    print("结果分布:", s["outcomes"])
    if s["stop_reasons"]:
        print("停止原因:", s["stop_reasons"])
    t = s["track_err_rad"]
    print(f"跟随误差（每回合最大值）中位 {t['p50']:.2f}、95% {t['p95']:.2f}、最大 {t['max']:.2f} rad")
    print(f"每拍计算中位 {s['compute_ms_median']:.0f} ms；目标修正：指尖抬高 {s['episodes_with_finger_lift']} 个回合，"
          f"截到软限位 {s['episodes_with_limit_clip']} 个回合")


def main(argv=None):
    ap = argparse.ArgumentParser(description="闭环干跑：真机闭环代码 + MuJoCo 虚拟实验室（不连任何硬件）")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--config", default=str(TRAIN_CONFIG))
    ap.add_argument("--profile", choices=sorted(PROFILES), default="sim", help="安全档位：sim 同训练；follow / grasp 是真机档位")
    ap.add_argument("--speed-scale", type=float, default=None, help="覆盖档位里的速度比例")
    ap.add_argument("--finger-clearance-mm", type=float, default=None, help="覆盖档位里指尖离安全带面的最小余量（mm）")
    ap.add_argument("--no-viewer", dest="viewer", action="store_false", help="不打开 MuJoCo 图形界面")
    ap.add_argument("--playback", type=float, default=1.0, help="图形界面播放倍速：1 实时，0 不等待")
    ap.add_argument("--frames", action="store_true", help="另存深度帧（step1 录制格式，可用 step2 回放）")
    ap.add_argument("--cam-latency-ms", type=float, default=0.0, help="仿真相机延迟：画面是这么久之前的场景（真机实测 80）")
    ap.add_argument("--latency-comp-ms", type=float, default=None, help="感知按多少延迟补偿（默认 = --cam-latency-ms，即补偿正好）")
    ap.add_argument("--latency-jitter-ms", type=float, default=0.0, help="每帧实际延迟的随机抖动标准差（ms）")
    ap.add_argument("--depth-noise-mm", type=float, default=0.0, help="0.3 m 处每像素深度噪声标准差（mm），按距离平方放大")
    ap.add_argument("--edge-dropout", type=float, default=0.0, help="深度跳变边缘像素置为无效的概率")
    ap.add_argument("--lab-belt", action="store_true",
                    help="安全带面按真机算：max(配置, 标定 step5 实测)，同 step6/7（默认按仿真带面）")
    args = ap.parse_args(argv)
    if args.viewer:
        from ...envs.viewer import display_available
        args.viewer = display_available()
    return run(args)


if __name__ == "__main__":
    main()
