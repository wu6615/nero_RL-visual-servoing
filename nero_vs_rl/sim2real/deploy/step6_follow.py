"""Step 6 (lab, arm moves with --execute) - policy in closed loop, but follow only: slow, gripper locked open, TCP >= 10 cm above the belt.

  python -m nero_vs_rl.sim2real.deploy.step6_follow                       # 只读预演：机械臂不动，看策略"想怎么走"
  python -m nero_vs_rl.sim2real.deploy.step6_follow --execute --episodes 3

每回合：开带、在上游放一个物体 → 回车 → 等第一次完整检出 → 策略跟随（不抓）→ 停住 → 回起点。
step7_grasp 用同一个流程，只是换成抓取档位。
"""
from __future__ import annotations

import argparse
import dataclasses
import time

from ... import REAL_DATA_DIR
from ...utils.config import load_config
from .lab import Preview, add_lab_args, ask, between_episodes, confirm_motion, open_lab, preflight, safe_belt_z
from .runtime.run_log import RunLog
from .runtime.safety import PROFILES
from .runtime.session import LoopSession, load_policy
from .step2_replay import DEFAULT_MODEL

# =============================================================================
# 【说明】真机闭环的一次运行（step6 跟随 / step7 抓取共用）：
#   开设备 → 开机检查（只读）→ [执行时] 输入 yes 确认 → 建 LoopSession（从这一刻起 100 Hz 发送"原地保持"）
#   → 每回合：操作者放物体、回车 → PolicyEpisode.run()（等目标 → 策略 → 停住）→ 回合间菜单
#   → 退出：停止发送，保存日志，打印回放命令
#   只读预演（不加 --execute）时整条链路照跑：感知、策略、IK、安全检查、"本应发出"的关节指令都记日志，
#   但 CAN 发送被禁止，机械臂不动——所以是开环的（同 step2 回放），用来在动之前看策略的意图是否合理。
# =============================================================================
def run_lab(stage: str, argv=None):
    ap = argparse.ArgumentParser(description=f"真机闭环（{PROFILES[stage].title}）")
    add_lab_args(ap)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--speed-scale", type=float, default=None, help=f"覆盖档位速度比例（默认 {PROFILES[stage].speed_scale}）")
    ap.add_argument("--finger-clearance-mm", type=float, default=None,
                    help=f"覆盖指尖离安全带面的最小余量（mm，默认 {PROFILES[stage].finger_clearance_m * 1000:.0f}）")
    ap.add_argument("--wait-s", type=float, default=30.0, help="每回合最多等多久物体出现")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-frames", action="store_true", help="不存深度帧（只存逐拍日志）")
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.config)
    profile = PROFILES[stage]
    if args.speed_scale is not None:
        profile = dataclasses.replace(profile, speed_scale=args.speed_scale)
    if args.finger_clearance_mm is not None:
        profile = dataclasses.replace(profile, finger_clearance_m=args.finger_clearance_mm / 1000)
    mode = "执行" if args.execute else "只读预演（机械臂不会动）"
    print(f"配置: {args.config}\n模型: {args.model}\n档位: {profile.name}（{profile.title}）\n模式: {mode}")

    dev = open_lab(args, need_camera=True)
    session = log = preview = None
    try:
        print("\n开机检查：")
        if preflight(dev, cfg):
            print("\n检查没通过，未启动。")
            return None
        belt_z = safe_belt_z(cfg, args.config)
        if args.execute:
            if not confirm_motion(profile, [f"安全带面 z = {belt_z:.3f} m（配置 {cfg.belt_surface_z:.3f}）；相机延迟 {cfg.camera.latency_s:.2f} s"]):
                print("没有确认，未启动。")
                return None
            dev.arm.prepare_motion()
        policy = load_policy(args.model)
        out = REAL_DATA_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_closed_loop_{stage}"
        log = RunLog(out, dict(source="real", hz=cfg.control_freq, config=str(args.config), model=args.model,
                               profile=dataclasses.asdict(profile), executed=args.execute,
                               camera=dev.camera.camera.camera_info(), created=time.strftime("%Y-%m-%d %H:%M:%S")),
                     save_frames=not args.no_frames)
        session = LoopSession(cfg, profile, dev.arm, dev.camera, dev.clock, belt_z_safe=belt_z, latency_s=cfg.camera.latency_s)
        preview = None if args.no_preview else Preview(dev.camera.K, camera=dev.camera)
        q_start = dev.arm.joints()
        for i in range(args.episodes):
            if ask(f"\n[回合 {i + 1}/{args.episodes}] 开带、在上游放好物体后按回车开始（q 退出）：", preview).strip().lower() == "q":
                break
            res = session.episode(policy, on_tick=preview, log=log, episode_index=i,
                                  wait_s=args.wait_s).run()
            log.episode(dict(episode=i, outcome=res.outcome, reasons=res.reasons, policy_steps=res.policy_steps,
                             t_start_servo=res.t_start_servo, t_lifted=res.t_lifted))
            print(f"结果：{res.outcome}，策略 {res.policy_steps} 步" + (f"；{'；'.join(res.reasons)}" if res.reasons else ""))
            if res.outcome == "estop":
                print("机械臂报故障，已急停。处理好故障后重新运行。")
                break
            if i + 1 < args.episodes and not between_episodes(session, q_start, preview):
                break
        print(f"\n日志：{out}\n离线回放（看策略当时看到了什么）：\n"
              f"python -m nero_vs_rl.sim2real.deploy.step2_replay --record {out} --video")
        return out
    finally:
        if session is not None:
            session.close()
        if log is not None:
            log.close()
        if preview is not None:
            preview.close()
        dev.close()


def main(argv=None):
    return run_lab("follow", argv)


if __name__ == "__main__":
    main()
