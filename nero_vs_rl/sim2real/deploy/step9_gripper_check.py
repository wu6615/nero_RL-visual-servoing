"""Step 9 (lab, gripper moves with --execute) - gripper timing, reading vs real finger gap (ruler), reading on held objects.

  python -m nero_vs_rl.sim2real.deploy.step9_gripper_check --sim        # 仿真干跑：只有开合计时（没有尺子和工件）
  python -m nero_vs_rl.sim2real.deploy.step9_gripper_check              # 真机只读预演
  python -m nero_vs_rl.sim2real.deploy.step9_gripper_check --execute    # 真机：机械臂不动，只动夹爪

输出：REAL_DATA_DIR/<时间>_gripper_check/report.json。
"""
from __future__ import annotations

import argparse

import numpy as np

from ...utils.config import load_config
from .common import write_json
from .lab import add_measure_args, open_measure_run
from .runtime.devices import q_to_width, width_to_q
from .runtime.safety import PROFILES

# =============================================================================
# 【说明】三部分，机械臂全程停在等待位不动：
#   A. 开合计时（自动）：全开 → 全闭、全闭 → 全开各 3 次，100 Hz 记读数。
#      算：开始动的延迟（变化 > 1 mm）、走完一半的时间、到终值 2 mm 内的时间、终值。
#      用来定仿真的夹爪驱动（MuJoCo kp 300 / kv 10；Isaac 的 gripper_stiffness / gripper_damping）。
#   B. 尺子（真机，要操作者）：指令 10 / 8 / 6 / 4 / 2 cm，每次停稳后用尺子量两指内侧间距，输入毫米。
#      拟合"实际间距 = a × 读数 + b"。仿真里指垫间距 = 0.008 + 2q，读数按 2q 换算（runtime/devices.py），
#      这一步核对两者差多少。
#   C. 夹工件（真机，要操作者）：操作者把工件放在两指之间扶住，回车后夹爪闭合，记夹住后的读数；
#      再输入卡尺量的工件宽度。用来核对"夹住时读数 ≈ 工件宽度 + 常数"，GraspMonitor 和 v8 的"真正夹住"都靠它。
#      夹爪力用 NERO 工程现场值 2 N（--gripper-force），手指夹到手不会伤人，但手指不要放在两指之间。
# =============================================================================
TRIALS = 3
WIDTHS = (0.10, 0.08, 0.06, 0.04, 0.02)


def sample(run, seconds: float, command=None) -> tuple[np.ndarray, np.ndarray]:
    """Gripper reading at 100 Hz for `seconds`; `command()` is sent right after the first reading."""
    clock, t0 = run.clock, run.clock.now()
    ts, ws = [0.0], [run.arm.gripper_width()]                 # 第一个样本 = 发指令之前
    if command is not None:
        command()
    for k in range(1, int(round(seconds / 0.01)) + 1):
        clock.sleep_until(t0 + k * 0.01)
        ts.append(clock.now() - t0)
        ws.append(run.arm.gripper_width())
    return np.array(ts), np.array(ws)


def timing(ts: np.ndarray, ws: np.ndarray) -> dict:
    w0, w1 = ws[0], ws[-1]
    moved = np.flatnonzero(np.abs(ws - w0) > 0.001)
    half = np.flatnonzero(np.abs(ws - w0) >= abs(w1 - w0) / 2)
    near = np.flatnonzero(np.abs(ws - w1) > 0.002)
    return dict(start_s=float(ts[moved[0]]) if len(moved) else float("nan"),
                half_s=float(ts[half[0]]) if len(half) and abs(w1 - w0) > 0.002 else float("nan"),
                done_s=float(ts[min(near[-1] + 1, len(ts) - 1)]) if len(near) else 0.0,
                from_m=float(w0), to_m=float(w1))


def ask_mm(prompt: str) -> float | None:
    s = input(prompt).strip()
    try:
        return float(s) / 1000 if s else None
    except ValueError:
        print("  不是数字，跳过")
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="夹爪：开合计时、读数与实际间距、夹工件时的读数")
    add_measure_args(ap)
    ap.add_argument("--skip-ruler", action="store_true", help="不做 B（尺子）")
    ap.add_argument("--skip-objects", action="store_true", help="不做 C（夹工件）")
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.config)
    run = open_measure_run(args, cfg, PROFILES["grasp"], "gripper_check",
                           ["动作：机械臂不动；夹爪全开 / 全闭来回 3 次，再按提示开到各个宽度、夹工件"])
    if run is None:
        return None
    report = dict(mode=run.mode, timing=[], ruler=[], objects=[])
    interactive = run.mode == "execute"
    try:
        s = run.session
        print("\nA. 开合计时：")
        s.set_gripper(0.10)
        sample(run, 1.0)
        for i in range(TRIALS):
            for name, w in (("闭合", 0.0), ("张开", 0.10)):
                ts, ws = sample(run, 1.2, lambda w=w: s.set_gripper(w))
                r = dict(timing(ts, ws), kind=name, trial=i, t=ts.tolist(), w=ws.tolist())
                report["timing"].append(r)
                print(f"  {name} {i + 1}: 开始动 {r['start_s'] * 1000:.0f} ms，过半 {r['half_s'] * 1000:.0f} ms，"
                      f"到位 {r['done_s'] * 1000:.0f} ms，{r['from_m'] * 100:.1f} → {r['to_m'] * 100:.1f} cm")
        if run.mode == "sim":
            print("\n仿真：指令宽度 → 读数 → 模型指垫间距（0.008 + 2q）")
            for w in WIDTHS:
                s.set_gripper(w)
                sample(run, 1.2)
                r = run.arm.gripper_width()
                print(f"  {w * 100:4.1f} cm → 读数 {r * 100:5.2f} cm → 指垫间距 {(0.008 + 2 * width_to_q(r)) * 100:5.2f} cm")
                report["ruler"].append(dict(cmd_m=w, reading_m=r, gap_m=0.008 + 2 * width_to_q(r), source="model"))
        if interactive and not args.skip_ruler:
            print("\nB. 尺子：每次停稳后量两指内侧最窄处的间距（毫米），回车跳过这一档")
            for w in WIDTHS:
                s.set_gripper(w)
                sample(run, 1.5)
                r = run.arm.gripper_width()
                gap = ask_mm(f"  指令 {w * 100:.0f} cm，读数 {r * 100:.2f} cm。实际间距（mm）：")
                report["ruler"].append(dict(cmd_m=w, reading_m=r, gap_m=gap, source="ruler"))
            pts = [(x["reading_m"], x["gap_m"]) for x in report["ruler"] if x["gap_m"] is not None]
            if len(pts) >= 2:
                a, b = np.polyfit(*np.array(pts).T, 1)
                report["ruler_fit"] = dict(a=float(a), b_m=float(b))
                print(f"  实际间距 ≈ {a:.3f} × 读数 + {b * 1000:.1f} mm（仿真模型：1.000 × 读数 + 8.0 mm）")
        if interactive and not args.skip_objects:
            print("\nC. 夹工件：手指不要放在两指之间；回车开始，q 结束")
            k = 0
            while input(f"  工件 {k + 1}：张开后把工件放到两指之间扶住，回车夹紧（q 结束）：").strip().lower() != "q":
                s.set_gripper(0.10)
                sample(run, 1.0)
                input("  放好后回车：")
                ts, ws = sample(run, 1.5, lambda: s.set_gripper(0.0))
                held = float(np.median(ws[-20:]))
                width = ask_mm(f"  夹住后读数 {held * 100:.2f} cm。卡尺量的工件宽度（mm）：")
                name = input("  工件编号或名称（可空）：").strip()
                report["objects"].append(dict(name=name, reading_m=held, object_m=width, t=ts.tolist(), w=ws.tolist()))
                s.set_gripper(0.10)
                sample(run, 1.0)
                k += 1
            pairs = [(o["object_m"], o["reading_m"]) for o in report["objects"] if o["object_m"]]
            if pairs:
                d = np.array([r - w for w, r in pairs])
                print(f"  读数 − 工件宽度：中位 {np.median(d) * 1000:+.1f} mm，范围 {d.min() * 1000:+.1f}～{d.max() * 1000:+.1f} mm")
        s.set_gripper(0.10)
        write_json(run.out / "report.json", report)
        print(f"报告：{run.out / 'report.json'}")
        return report
    finally:
        run.close()


if __name__ == "__main__":
    main()
