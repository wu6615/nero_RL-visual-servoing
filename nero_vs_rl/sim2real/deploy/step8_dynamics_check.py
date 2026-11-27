"""Step 8 (lab, arm moves with --execute) - fast TCP moves (10-50 cm/s): lag, overshoot, settling, sag. No policy.

  python -m nero_vs_rl.sim2real.deploy.step8_dynamics_check --sim                 # 仿真干跑（打开 MuJoCo 图形界面）
  python -m nero_vs_rl.sim2real.deploy.step8_dynamics_check                       # 真机只读预演
  python -m nero_vs_rl.sim2real.deploy.step8_dynamics_check --execute             # 真机

输出：REAL_DATA_DIR/<时间>_dynamics_check/report.json（每段的结果）+ joints.npz（100 Hz 实测关节和发出的指令）。
"""
from __future__ import annotations

import argparse

import numpy as np

from ...utils.config import load_config
from .common import write_json
from .lab import add_measure_args, open_measure_run
from .runtime.safety import PROFILES
from .runtime.scripted import ScriptedMotion, best_lag

# =============================================================================
# 【说明】为什么要测：step5 只走 3 cm/s，而策略抓取时 TCP 跑到过 40～60 cm/s；上次真机比仿真"多冲 2～3 cm"，
#   要知道真机在这个速度下的滞后、刹车距离、停稳时间是多少，Isaac 的关节驱动参数（isaac.yaml 的 drive）就按它调，
#   MuJoCo 的执行器也拿它核对。
#   动作：在等待位附近（离带面 25 cm 以上），沿 y（带向）、x（横向）、z（竖直）各走一个来回，
#     速度 10 / 20 / 35 / 50 cm/s，距离 8 cm（z 向上 6 cm）；每段末尾停 0.8 s。另加一个"一拍跳 5 cm"。
#   每段算：
#     滞后      实测 TCP 相对"每拍指令 TCP"（10 Hz 目标，发送器用 0.1 s 插值过去）的最佳时间平移，沿运动方向；
#               比相对 100 Hz 插值指令的滞后多约 0.05～0.1 s，比较时用同一个口径
#     超调      停下后实测越过终点的最大距离（沿运动方向）
#     停稳      进入终点 2 mm 以内所需时间（从指令到终点算起）
#     稳态误差  停 0.8 s 后实测与终点的距离（含重力下沉）
#     峰值速度  实测 TCP 的最大速度
#   档位用 grasp（速度 ×1.0）：夹爪不动，TCP 始终在带面上 20 cm 以上，带面和护栏碰不到。
# =============================================================================
SPEEDS = (0.10, 0.20, 0.35, 0.50)
AXES = (("y", np.array([0.0, 1.0, 0.0]), 0.08), ("x", np.array([1.0, 0.0, 0.0]), 0.08), ("z", np.array([0.0, 0.0, 1.0]), 0.06))
PAUSE_S = 0.8


def plan() -> list[tuple[str, np.ndarray, float]]:
    """(name, displacement, speed) in order; every out-and-back returns to the start."""
    out = []
    for ax, u, d in AXES:
        for v in SPEEDS:
            out += [(f"{ax} +{d * 100:.0f} cm @ {v * 100:.0f} cm/s", u * d, v), (f"{ax} 回 @ {v * 100:.0f} cm/s", -u * d, v)]
    out += [("y 一拍跳 +5 cm", np.array([0.0, 0.05, 0.0]), 0.5), ("y 一拍跳回", np.array([0.0, -0.05, 0.0]), 0.5)]
    return out


def analyse(mo: ScriptedMotion, segments: list[dict]) -> list[dict]:
    """Per segment numbers from the 100 Hz measured joints and the 10 Hz commanded TCP."""
    a = mo.arrays()
    if len(a["t_meas"]) < 10:
        return []
    tm = a["t_meas"]
    pm = np.array([mo.ik.fk(q)[0] for q in a["q_meas"]])
    tc = np.array([r["t"] for r in mo.rows])                                # 走和停的每一拍：停的时候指令就是终点
    pc = np.array([r["p_cmd"] for r in mo.rows])
    out = []
    for seg in segments:
        u = np.asarray(seg["disp"]) / np.linalg.norm(seg["disp"])
        end, t_end = np.asarray(seg["p_end"]), seg["t_cmd_end"]
        win = (tm >= seg["t_start"]) & (tm <= seg["t_hold_end"])
        cw = (tc >= seg["t_start"] - 1e-9) & (tc <= seg["t_hold_end"] + 1e-9)
        cw[max(int(np.argmax(cw)) - 1, 0)] = True                              # 带上段前一拍（起点）
        if win.sum() < 5 or cw.sum() < 2:
            continue
        lag, rms = best_lag(tc[cw], pc[cw] @ u, tm[win], pm[win] @ u)
        after = win & (tm >= t_end)
        along = (pm[after] - end) @ u
        dist = np.linalg.norm(pm[after] - end, axis=1)
        inside = np.flatnonzero(dist < 0.002)
        settle = float(tm[after][inside[0]] - t_end) if len(inside) else float("nan")
        v = np.linalg.norm(np.diff(pm[win], axis=0), axis=1) / np.maximum(np.diff(tm[win]), 1e-4)
        out.append(dict(name=seg["name"], speed_cmd=seg["speed"], lag_s=lag, rms_m=rms,
                        overshoot_m=float(max(along.max(), 0.0)) if len(along) else float("nan"),
                        settle_s=settle, final_err_m=float(dist[-1]) if len(dist) else float("nan"),
                        sag_z_m=float(pm[after][-1, 2] - end[2]) if len(dist) else float("nan"),
                        peak_speed=float(v.max()) if len(v) else float("nan")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="快速运动响应：滞后、超调、停稳、下沉（不用策略）")
    add_measure_args(ap)
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.config)
    moves = plan()
    run = open_measure_run(args, cfg, PROFILES["grasp"], "dynamics_check",
                           [f"动作：等待位附近沿 y/x/z 来回，{min(SPEEDS) * 100:.0f}～{max(SPEEDS) * 100:.0f} cm/s，"
                            f"最远离开 8 cm，共 {len(moves)} 段；夹爪不动"])
    if run is None:
        return None
    report = dict(mode=run.mode, segments=[], problems=[])
    try:
        mo = ScriptedMotion(run.session)
        q_home = run.session.streamer.q_cmd.copy()
        segs = []
        print("\n动作：")
        for name, disp, v in moves:
            t0, p_end = run.clock.now(), mo.p + disp
            print(f"  {name} …", end="", flush=True)
            problems = mo.line(p_end, speed=v, name=name)
            t_cmd_end = run.clock.now()
            problems = problems or mo.hold(PAUSE_S, name + " 停")
            if problems:
                print(" 停止：" + "；".join(problems))
                report["problems"] = problems
                break
            segs.append(dict(name=name, disp=disp, speed=v, p_end=p_end, t_start=t0, t_cmd_end=t_cmd_end, t_hold_end=run.clock.now()))
            print(" 完成")
        if not report["problems"]:
            mo.home(q_home)
        report["segments"] = analyse(mo, segs) if run.arm.can_move else []
        if report["segments"]:
            print(f"\n{'段':28s}{'滞后ms':>8s}{'超调mm':>8s}{'停稳ms':>8s}{'稳态mm':>8s}{'下沉mm':>8s}{'峰速cm/s':>10s}")
            for r in report["segments"]:
                print(f"{r['name']:28s}{r['lag_s'] * 1000:8.0f}{r['overshoot_m'] * 1000:8.1f}{r['settle_s'] * 1000:8.0f}"
                      f"{r['final_err_m'] * 1000:8.1f}{r['sag_z_m'] * 1000:8.1f}{r['peak_speed'] * 100:10.0f}")
        elif not run.arm.can_move and not report["problems"]:
            print(f"\n只读预演：{len(mo.rows)} 拍指令都通过了安全检查")
        report["ticks"] = mo.rows
        np.savez_compressed(run.out / "joints.npz", **mo.arrays())
        write_json(run.out / "report.json", report)
        print(f"报告：{run.out / 'report.json'}")
        return report
    finally:
        run.close()


if __name__ == "__main__":
    main()
