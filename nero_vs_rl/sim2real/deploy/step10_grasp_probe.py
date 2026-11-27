"""Step 10 (lab, arm moves with --execute) - scripted grasps on a static workpiece at known offsets / angles / speeds. No policy.

  python -m nero_vs_rl.sim2real.deploy.step10_grasp_probe --sim                       # 仿真干跑（默认工件：6.0×8.2×5.9 cm 接线盒）
  python -m nero_vs_rl.sim2real.deploy.step10_grasp_probe                             # 真机只读预演
  python -m nero_vs_rl.sim2real.deploy.step10_grasp_probe --execute                   # 真机：带停住，工件放在相机画面中间

真机跑完后，加 --sim --replay-real 和那次运行的目录（REAL_DATA_DIR 下的 *_grasp_probe）：同一个工件、同一个起始关节、
同一组试次搬进仿真，跑完打印"真机 / 仿真"对照表。真机运行结束时会打印这条命令。
输出：REAL_DATA_DIR/<时间>_grasp_probe/report.json + joints.npz。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...utils.config import load_config
from ...utils.geometry import make_T, rot_z
from .common import write_json
from .lab import add_measure_args, open_measure_run
from .runtime.episode import LivePerception
from .runtime.safety import PROFILES
from .runtime.scripted import ScriptedMotion

# =============================================================================
# 【说明】要回答的问题：同样的"手相对工件的偏差 / 转角 / 速度"，真机和仿真夹不夹得住、工件被推开多远？
#   上次分析卡在这里：真机闭合时手斜着、还在动，全部夹空；仿真同样的姿态却能"夹住"（工件斜卡在两指间）。
#   每个试次：
#     1. 在等待位用真机感知定位静止工件（10 帧中位数：中心、朝向、尺寸）；
#     2. 用解析抓取（不倾斜、闭合方向跨过窄边、居中）作基准位姿 G，再按试次加偏差：
#          d_close 沿闭合方向（TCP y）、d_along 沿两指连线的垂直方向（TCP x）、dyaw 绕竖直轴转；
#     3. 走到 G 上方 10 cm（15 cm/s）→ 降到 G（8 cm/s）→ 停 0.4 s → 合拢 → 0.8 s 后记开口 → 抬 8 cm → 记开口；
#        "边走边合"的试次：在 G 后方 8 cm（沿 TCP x）降到抓取高度，以 v 穿过 G，过 G 的那一拍合拢；
#     4. 张开、回等待位，重新定位工件，算被推开多远；操作者把工件放回原处再做下一个。
#   判定：夹住 = 抬起后开口 ≥ 1.5 cm（两指间有东西）；真机另由操作者回答 y/n/s（s = 夹住但歪了或滑了）。
#   安全：档位 grasp（指尖不低于安全带面 8 mm，速度 ≤ 30 cm/s）；张开的手指如果会落在工件顶面上，跳过该试次
#   （跨 6 cm 窄边、全开 10.8 cm 时每侧只剩约 1.4 cm，所以闭合方向的偏差只试到 1.5 cm）。
# =============================================================================
TRIALS = [  # 名称, d_close (m), d_along (m), dyaw (°), 边走边合速度 (m/s，0 = 停稳再合)
    ("正对", 0.0, 0.0, 0, 0.0),
    ("闭合方向偏 0.5 cm", 0.005, 0.0, 0, 0.0), ("闭合方向偏 1 cm", 0.01, 0.0, 0, 0.0), ("闭合方向偏 1.5 cm", 0.015, 0.0, 0, 0.0),
    ("闭合方向偏 −1 cm", -0.01, 0.0, 0, 0.0),
    ("沿长边偏 2 cm", 0.0, 0.02, 0, 0.0), ("沿长边偏 4 cm", 0.0, 0.04, 0, 0.0),
    ("转 15°", 0.0, 0.0, 15, 0.0), ("转 30°", 0.0, 0.0, 30, 0.0), ("转 45°", 0.0, 0.0, 45, 0.0),
    ("边走边合 10 cm/s", 0.0, 0.0, 0, 0.10), ("边走边合 20 cm/s", 0.0, 0.0, 0, 0.20), ("边走边合 30 cm/s", 0.0, 0.0, 0, 0.30),
]
OPEN_GAP_M = 0.108                # 夹爪全开时的指垫间距（仿真模型；step9 用尺子核对）
PAD_HALF_M, MARGIN_M = 0.006, 0.004
HELD_MIN_M = 0.015
LIFT_M = 0.08                     # 比成功判据的 6 cm 多抬 2 cm，仿真真值不会卡在判定线上
DEFAULT_SIM_OBJECT = dict(obj_type="box", half=[0.030, 0.041, 0.0295], center=[-0.390, 0.120, -0.0845], yaw=np.radians(-3))


# ------------------------------------------------------------ 工件
def locate(run, perception, n: int = 10, timeout: float = 8.0) -> dict | None:
    """Median of n usable detections of the static workpiece (arm still)."""
    clock, t0, dets = run.clock, run.clock.now(), []
    perception.reset()
    while len(dets) < n and clock.now() - t0 < timeout:
        clock.sleep_until(clock.now() + 0.1)
        seen = perception.update(run.camera.latest(), run.arm, clock.now() - t0)
        if seen.usable and seen.det is not None:
            dets.append(seen.det)
    if len(dets) < n // 2:
        return None
    types = [d.obj_type for d in dets]
    kind = max(set(types), key=types.count)
    dets = [d for d in dets if d.obj_type == kind]
    period = np.pi if kind == "box" else 2 * np.pi                  # 长方体朝向差 180° 是同一个
    yaw = float(np.angle(np.mean(np.exp(1j * np.array([d.yaw for d in dets]) * 2 * np.pi / period))) * period / (2 * np.pi))
    return dict(obj_type=kind, center=np.median([d.center for d in dets], 0).tolist(), yaw=yaw,
                half=np.median([d.half for d in dets], 0).tolist(), n=len(dets))


def base_grasp(obj: dict, detector, R_now: np.ndarray) -> np.ndarray:
    """Untilted centred grasp across the narrow side (box) or the jaw direction closest to now (cylinder); 4x4 TCP pose."""
    from .perception import Detection
    det = Detection(obj["obj_type"], np.asarray(obj["center"]), obj["yaw"], np.asarray(obj["half"]), 0, False, 1.0, np.zeros(3))
    gs = detector.grasps(det, 0.0)
    T_wo = make_T(rot_z(obj["yaw"]), np.asarray(obj["center"]))
    best, cost = None, np.inf
    for T in gs.T_obj:
        G = T_wo @ T
        for flip in (np.eye(3), np.diag([-1.0, -1.0, 1.0])):        # 二指对称：绕接近方向转 180° 是同一个抓取
            Gf = G.copy()
            Gf[:3, :3] = G[:3, :3] @ flip
            width = 2 * np.abs(np.asarray(obj["half"])[:2] @ np.abs(T[:2, 1]))
            c = np.linalg.norm(T[:2, 3]) * 10 + width * 30 + np.arccos(np.clip((np.trace(R_now.T @ Gf[:3, :3]) - 1) / 2, -1, 1)) * 0.1
            if c < cost:
                best, cost = Gf, c
    return best


def trial_pose(G: np.ndarray, d_close: float, d_along: float, dyaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    R = G[:3, :3] @ rot_z(np.radians(dyaw_deg))
    return G[:3, 3] + G[:3, :3] @ np.array([d_along, d_close, 0.0]), R


def finger_on_top(obj: dict, p: np.ndarray, R: np.ndarray) -> bool:
    """Would an open fingertip pad come down onto the workpiece's top face at this TCP pose?"""
    half = np.asarray(obj["half"])
    for s in (-1.0, 1.0):
        c = p + R @ np.array([0.0, s * OPEN_GAP_M / 2, 0.0])
        local = rot_z(obj["yaw"]).T @ (c - np.asarray(obj["center"]))
        if obj["obj_type"] == "box":
            inside = np.all(np.abs(local[:2]) < half[:2] + PAD_HALF_M + MARGIN_M)
        else:
            inside = np.hypot(*local[:2]) < half[0] + PAD_HALF_M + MARGIN_M
        if inside:
            return True
    return False


# ------------------------------------------------------------ 一个试次
def run_trial(mo: ScriptedMotion, G: np.ndarray, spec: tuple, obj: dict) -> dict:
    name, d_close, d_along, dyaw, v = spec
    p, R = trial_pose(G, d_close, d_along, dyaw)
    x = R[:, 0]
    start = p - x * 0.08 if v > 0 else p
    out = dict(name=name, d_close=d_close, d_along=d_along, dyaw_deg=dyaw, speed=v, target=p.tolist())
    for q in (start, p):
        if finger_on_top(obj, q, R):
            return dict(out, skipped="张开的手指会落在工件顶面上")
    above = start + np.array([0.0, 0.0, 0.10])
    mo.gripper(0.10)
    problems = mo.line(above, R, 0.15, name + " 到上方") or mo.line(start, R, 0.08, name + " 下降") or mo.hold(0.4, name + " 停稳")
    if not problems:
        pm, _ = mo.measured_tcp()
        out["reach_err_m"] = float(np.linalg.norm(pm - start))
        if v > 0:
            n_cross = {}

            def close_at_g(k, n, pc):
                if "k" not in n_cross and (pc - p) @ x >= 0:
                    n_cross["k"] = k
                    mo.gripper(0.0)
            problems = mo.line(p + x * 0.03, R, v, name + " 边走边合", on_tick=close_at_g)
        else:
            mo.gripper(0.0)
        problems = problems or mo.hold(0.8, name + " 合拢")
    if not problems:
        out["width_closed_m"] = mo.s.arm.gripper_width()
        problems = mo.line(mo.p + np.array([0.0, 0.0, LIFT_M]), None, 0.05, name + " 抬起") or mo.hold(0.5, name + " 抬起停")
        out["width_lifted_m"] = mo.s.arm.gripper_width()
        out["held"] = bool(out["width_lifted_m"] >= HELD_MIN_M)
    out["problems"] = problems
    return out


# ------------------------------------------------------------ 主程序
def main(argv=None):
    ap = argparse.ArgumentParser(description="脚本抓取探针：已知偏差 / 转角 / 速度下夹不夹得住（不用策略）")
    add_measure_args(ap)
    ap.add_argument("--trials", default=None, help="只做这些试次（序号，逗号分隔，从 0 开始）；默认全部")
    ap.add_argument("--replay-real", default=None, help="仿真里复现一次真机 step10 运行（它的 *_grasp_probe 目录）")
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.config)
    real = json.loads((Path(args.replay_real) / "report.json").read_text(encoding="utf-8")) if args.replay_real else None
    if real is not None and not args.sim:
        raise SystemExit("--replay-real 只能和 --sim 一起用")
    idx = [int(i) for i in args.trials.split(",")] if args.trials else list(range(len(TRIALS)))
    specs = [TRIALS[i] for i in idx] if real is None else [tuple(t["spec"]) for t in real["trials"]]
    sim_obj = (real["object"] if real is not None else DEFAULT_SIM_OBJECT) if args.sim else None
    sim_cfg = place = None
    if args.sim:
        from .reproduce_in_sim import world_config
        h, c = np.asarray(sim_obj["half"]), np.asarray(sim_obj["center"])
        sim_cfg = world_config(cfg, dict(sim_obj, half=h, center=c, belt_z=float(c[2] - h[2])), None)

        def place(cell):
            env = cell.env
            env.data.eq_active[env.obj["eq"]], env._attached = 1, True             # 重新挂上驱动 = 静止摆好
            cell.place_static_object(float(c[0]), float(c[1]), float(sim_obj["yaw"]))
            if real is not None:
                cell.set_arm(np.asarray(real["start_joints"]))
    run = open_measure_run(args, cfg, PROFILES["grasp"], "grasp_probe",
                           [f"动作：{len(specs)} 个脚本抓取试次（停稳再合 / 边走边合 ≤ 30 cm/s），每次抬起 8 cm 再放下回等待位",
                            "带要停住；每个试次之后按提示把工件放回原处"], need_camera=True, sim_setup=place, sim_cfg=sim_cfg)
    if run is None:
        return None
    report = dict(mode=run.mode, trials=[], replay_of=args.replay_real)
    try:
        s = run.session
        perception = LivePerception(cfg, s.twin, run.camera.K, run.camera.D, s.latency_s)
        mo = ScriptedMotion(s)
        q_home = s.streamer.q_cmd.copy()
        report["start_joints"] = q_home.tolist()
        obj = real["object"] if real is not None else locate(run, perception)
        if obj is None:
            print("没有完整检出工件：放在相机画面中间、带停住后重跑")
            return None
        report["object"] = obj
        print(f"\n工件：{obj['obj_type']} 尺寸 {np.round(np.asarray(obj['half']) * 200, 1).tolist()} cm，"
              f"中心 {np.round(obj['center'], 3).tolist()}，朝向 {np.degrees(obj['yaw']):.0f}°")
        for k, spec in enumerate(specs):
            ref = obj if real is not None or k == 0 else (locate(run, perception) or obj)
            G = base_grasp(ref, perception.detector, mo.R)
            print(f"[{k + 1}/{len(specs)}] {spec[0]} …", end="", flush=True)
            r = dict(run_trial(mo, G, spec, ref), spec=list(spec), object_before=ref)
            if "skipped" in r:
                print(" 跳过：" + r["skipped"])
            else:
                held = r.get("held")
                if run.mode == "sim":
                    tr = run.cell.truth()
                    r["truth_lifted"] = bool(tr["lifted"])
                    r["truth_two_fingers"] = bool(run.cell.env._contacts["left"] and run.cell.env._contacts["right"])
                elif run.mode == "execute":
                    r["operator"] = input(f" 开口 {r.get('width_lifted_m', 0) * 100:.1f} cm。夹起来了吗？y 夹住 / n 没夹住 / s 夹住但歪了或滑了：").strip().lower()
                mo.gripper(0.10)
                mo.hold(0.8, spec[0] + " 张开")
                problems = r["problems"] or mo.home(q_home)
                if problems:
                    r["problems"] = problems
                    print(" 停止：" + "；".join(problems))
                    report["trials"].append(r)
                    break
                after = locate(run, perception)
                if after is not None:
                    r["object_after"] = after
                    r["pushed_m"] = float(np.linalg.norm(np.subtract(after["center"][:2], ref["center"][:2])))
                print(f" 合拢后开口 {r.get('width_closed_m', 0) * 100:.1f} cm，抬起后 {r.get('width_lifted_m', 0) * 100:.1f} cm，"
                      f"{'夹住' if held else '没夹住'}" + (f"（仿真真值 {'抬起' if r['truth_lifted'] else '没有'}）" if "truth_lifted" in r else "")
                      + (f"，工件移动 {r['pushed_m'] * 100:.1f} cm" if "pushed_m" in r else ""))
            report["trials"].append(r)
            if k + 1 < len(specs):
                if run.mode == "sim":
                    place(run.cell)
                    mo.home(q_home)
                elif run.mode == "execute":
                    input("  把工件放回原处（白框对齐）后回车：")
        if real is not None:
            print(f"\n{'试次':18s}{'真机':>8s}{'仿真':>8s}{'真机推开cm':>12s}{'仿真推开cm':>12s}")
            for a, b in zip(real["trials"], report["trials"]):
                f = lambda t: "跳过" if "skipped" in t else ("夹住" if t.get("held") else "没夹住")
                g = lambda t: f"{t['pushed_m'] * 100:.1f}" if "pushed_m" in t else "-"
                print(f"{a['spec'][0]:18s}{f(a):>8s}{f(b):>8s}{g(a):>12s}{g(b):>12s}")
        report["ticks"] = mo.rows
        np.savez_compressed(run.out / "joints.npz", **mo.arrays())
        write_json(run.out / "report.json", report)
        print(f"报告：{run.out / 'report.json'}")
        if run.mode == "execute":
            print(f"仿真对照：\npython -m nero_vs_rl.sim2real.deploy.step10_grasp_probe --sim --replay-real {run.out}")
        return report
    finally:
        run.close()


if __name__ == "__main__":
    main()
