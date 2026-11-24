"""Step 5 (lab, arm moves with --execute) - first motion: small scripted TCP moves through the real execution layer, no policy.

  python -m nero_vs_rl.sim2real.deploy.step5_motion_check --sim                 # 仿真干跑（打开 MuJoCo 图形界面）
  python -m nero_vs_rl.sim2real.deploy.step5_motion_check                       # 真机只读预演：算指令、过安全检查，不发
  python -m nero_vs_rl.sim2real.deploy.step5_motion_check --execute             # 真机：上 3 cm 再回来，沿带 ±3 cm 再回来
  python -m nero_vs_rl.sim2real.deploy.step5_motion_check --execute --gripper   # 另外试夹爪：闭到 5 cm 再张开

输出：每段动作的跟随误差、跟随延迟、100 Hz 发送的实际频率；结果存 REAL_DATA_DIR/<时间>_motion_check/report.json（--sim 时存 SIM_OUTPUT_DIR/motion_check_sim/）。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ... import REAL_DATA_DIR, SIM_OUTPUT_DIR
from ...utils.config import load_config
from .common import write_json
from .lab import add_lab_args, confirm_motion, open_lab, preflight, safe_belt_z
from .runtime.safety import PROFILES
from .runtime.session import LoopSession

# =============================================================================
# 【说明】为什么第一次让机械臂动不用策略：
#   要先单独确认"执行层"本身——同一份 IK → 安全修正与检查 → 100 Hz 插值 → move_js——在真机上表现正常：
#   动作方向对不对（基座系 +z 真的是向上）、跟随误差和延迟多大（决定 max_tracking_err_rad 的余量）、
#   发送线程能不能稳定 100 Hz。动作很小（±3 cm）、很慢（3 cm/s）、全在等待位附近、离带面 20 cm 以上。
#   每拍（10 Hz）的做法和策略闭环完全一样：期望 TCP 沿直线走一小段 → IK（单拍关节变化受档位限速）→
#   guard.project / check_target → streamer.set_target。档位用 follow（速度 ×0.3、夹爪不动、TCP 离带面 ≥ 10 cm）。
# 跟随延迟：在指令序列上找一个时间平移 τ，使"实测 ≈ τ 秒前的指令"误差最小（取动得最多的那个关节）。
# =============================================================================
MOVES = [("上 3 cm", [0, 0, 0.03]), ("回", [0, 0, -0.03]),
         ("沿带 +Y 3 cm", [0, 0.03, 0]), ("回", [0, -0.03, 0]),
         ("沿带 −Y 3 cm", [0, -0.03, 0]), ("回", [0, 0.03, 0])]
SPEED_M_S = 0.03
PAUSE_S = 0.5


def run_moves(session, log_rows: list) -> list[str]:
    tw, guard, clock, streamer, arm = session.twin, session.guard, session.clock, session.streamer, session.arm
    ik, dt = tw.controller.ik, tw.control_dt
    vmax = np.asarray(tw.cfg.max_joint_speed_rad_s, float)
    p_cmd, R0 = ik.fk(streamer.q_cmd)
    for name, dp in MOVES:
        dp = np.asarray(dp, float)
        n = int(np.ceil(np.linalg.norm(dp) / SPEED_M_S / dt))
        p_from = p_cmd.copy()
        print(f"  {name} …", end="", flush=True)
        t_seg = clock.now()
        for k in range(1, n + 1):
            clock.sleep_until(t_seg + k * dt)
            now = clock.now()
            problems = arm.faults() + guard.check_motion(now, arm.joints(), streamer, executing=arm.can_move)
            p_des = p_from + dp * k / n
            q_tgt, _, _ = ik.solve(streamer.target(), p_des, R0, iters=10, max_step=vmax * dt)
            q_tgt, fixes = guard.project(q_tgt, 0.05)
            problems += guard.check_target(streamer.target(), q_tgt, 0.05)
            log_rows.append(dict(move=name, t=now, q=arm.joints(), q_tgt=q_tgt, fixes=fixes, problems=problems))
            if problems:
                streamer.hold(arm.joints(), now)
                print(" 停止：" + "；".join(problems))
                return problems
            streamer.set_target(q_tgt, now, dt)
        p_cmd = p_from + dp
        clock.sleep_until(clock.now() + PAUSE_S)
        if arm.can_move:
            p_meas, _ = ik.fk(arm.joints())
            print(f" 到位误差 {np.linalg.norm(p_meas - p_cmd) * 1000:.1f} mm")
        else:
            print(" 指令已算出并通过检查（只读预演，未发送）")
    return []


def gripper_check(session) -> dict:
    arm, clock = session.arm, session.clock
    out = {}
    for w in (0.05, 0.10):
        session.set_gripper(w)
        clock.sleep_until(clock.now() + 1.5)
        out[f"cmd_{w:.2f}"] = arm.gripper_width()
        print(f"  夹爪指令 {w * 100:.0f} cm → 读数 {out[f'cmd_{w:.2f}'] * 100:.1f} cm")
    return out


def tracking_report(streamer, hist: list[tuple[float, np.ndarray]]) -> dict:
    """Tracking error and lag of measured joints against the commands actually streamed."""
    if not streamer.sent or len(hist) < 10:
        return {}
    ts = np.array([t for t, _ in streamer.sent])
    qs = np.array([q for _, q in streamer.sent])
    th = np.array([t for t, _ in hist])
    qh = np.array([q for _, q in hist])
    keep = (th >= ts[0]) & (th <= ts[-1])
    th, qh = th[keep], qh[keep]
    j = int(np.argmax(np.ptp(qs, axis=0)))                          # 动得最多的关节
    lags = np.arange(0.0, 0.301, 0.005)
    rms = [np.sqrt(np.mean((qh[:, j] - np.interp(th - lag, ts, qs[:, j])) ** 2)) for lag in lags]
    err = np.abs(qh - np.array([np.interp(th, ts, qs[:, i]) for i in range(7)]).T)
    dts = np.diff(ts)
    return dict(joint=j + 1, lag_s=float(lags[int(np.argmin(rms))]), rms_at_lag_rad=float(min(rms)),
                max_err_rad=float(err.max()), stream_hz=float(1.0 / np.median(dts)),
                stream_gap_max_ms=float(dts.max() * 1000), clamped=int(streamer.n_clamped), sent=int(streamer.n_sent))


def main(argv=None):
    ap = argparse.ArgumentParser(description="第一次运动：脚本小动作检查执行层（不用策略）")
    add_lab_args(ap)
    ap.add_argument("--sim", action="store_true", help="在 MuJoCo 虚拟实验室里跑（不连硬件）")
    ap.add_argument("--no-viewer", dest="viewer", action="store_false")
    ap.add_argument("--gripper", action="store_true", help="另外检查夹爪开口（闭到 5 cm 再张开）")
    ap.add_argument("--out-root", default=None, help="报告目录的上级（默认：真机 REAL_DATA_DIR，--sim 为 SIM_OUTPUT_DIR/motion_check_sim）")
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.config)
    profile = PROFILES["follow"]
    # 仿真干跑的报告放 SIM_OUTPUT_DIR/motion_check_sim/ 下（以前直接放 SIM_OUTPUT_DIR 顶层，每次测试都多一个目录）
    root = Path(args.out_root) if args.out_root else (SIM_OUTPUT_DIR / "motion_check_sim" if args.sim else REAL_DATA_DIR)
    out = root / f"{time.strftime('%Y%m%d_%H%M%S')}_motion_check"
    out.mkdir(parents=True, exist_ok=True)
    if args.sim:
        from ...envs.viewer import display_available
        from ..hardware.sim_workcell import SimWorkcell
        cell = SimWorkcell(cfg, viewer=args.viewer and display_available())
        cell.reset(seed=0, object_types=["box"])
        cell.park_object()                                                      # 物体挪到远处、带停：只看机械臂
        arm, clock, belt_z, closer = cell.arm, cell.clock, cfg.belt_surface_z, cell.close
    else:
        dev = open_lab(args, need_camera=False)
        arm, clock, closer = dev.arm, dev.clock, dev.close
        belt_z = safe_belt_z(cfg, args.config)
    session = None
    rows: list = []
    report = dict(mode="sim" if args.sim else ("execute" if args.execute else "rehearsal"))
    try:
        if not args.sim:
            print("\n开机检查：")
            if preflight(dev, cfg):
                print("\n检查没通过，未启动。")
                return None
            if args.execute:
                if not confirm_motion(profile, [f"动作：{'、'.join(m[0] for m in MOVES)}，{SPEED_M_S * 100:.0f} cm/s"]):
                    print("没有确认，未启动。")
                    return None
                arm.prepare_motion()
        session = LoopSession(cfg, profile, arm, None, clock, belt_z_safe=belt_z, latency_s=0.0)
        t0 = clock.now()
        print("\n动作：")
        report["problems"] = run_moves(session, rows)
        if args.gripper and not report["problems"]:
            print("\n夹爪：")
            report["gripper"] = gripper_check(session)
        hist = arm.history(t0) if hasattr(arm, "history") else []
        report["tracking"] = tracking_report(session.streamer, hist)
        tr = report["tracking"]
        if tr and arm.can_move:
            print(f"\n跟随（J{tr['joint']}）：延迟约 {tr['lag_s'] * 1000:.0f} ms，最大误差 {tr['max_err_rad']:.3f} rad；"
                  f"发送 {tr['stream_hz']:.0f} Hz（最长间隔 {tr['stream_gap_max_ms']:.0f} ms），截断 {tr['clamped']} 次")
        elif not arm.can_move and not report["problems"]:
            print(f"\n只读预演：{len(rows)} 拍指令都通过了安全检查")
        report["ticks"] = rows
        write_json(out / "report.json", report)
        print(f"报告：{out / 'report.json'}")
        return report
    finally:
        if session is not None:
            session.close()
        closer()


if __name__ == "__main__":
    main()
