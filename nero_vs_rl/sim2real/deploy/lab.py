"""Lab-side workflow shared by steps 5-7: open devices, pre-flight checks, operator confirmation, preview, between-episode menu."""
from __future__ import annotations

import argparse
import select
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ... import TRAIN_CONFIG
from .common import RobotTwin, measured_belt_z
from .runtime.devices import FrameGrabber, RealClock
from .runtime.safety import SafetyProfile
from .step1_record import depth_preview, pose_error, pose_ok

# =============================================================================
# 【说明】现场每一步（step5～7）开头都一样：
#   1. 开设备：DC1（取帧线程）+ NERO（NeroController；没加 --execute 时 CAN 发送被禁掉 = 只读预演）
#   2. 开机检查（全部只读）：机械臂无故障（执行时还要求 7 个关节和夹爪都已使能）、相机出帧、
#      起始位姿在训练的随机范围内（同 step1 --check-pose）、机械臂静止
#   3. 执行时：打印本档位的限值，操作者在终端输入 yes 才开始（手放在急停上）
#   回合之间由操作者在终端选：张开夹爪并回起点 / 只张开 / 只回起点 / 退出。
#   退出时发送器停止，机械臂停在最后一个位置（位置模式），不急停、不断使能。
# =============================================================================
@dataclass
class LabDevices:
    arm: object
    camera: object | None
    clock: object
    _dc1: object | None = None

    def close(self):
        self.clock.stop()
        if self.camera is not None:
            self.camera.stop()
        if self._dc1 is not None:
            self._dc1.stop()
        self.arm.close()


def add_lab_args(ap: argparse.ArgumentParser):
    ap.add_argument("--execute", action="store_true", help="真的让机械臂动（不加 = 只读预演，CAN 发送被禁止）")
    ap.add_argument("--config", default=str(TRAIN_CONFIG))
    ap.add_argument("--can", default="can0")
    ap.add_argument("--firmware", default="v121")
    ap.add_argument("--serial", default=None, help="DC1 序列号（多台时）")
    ap.add_argument("--joint-mode", choices=["js", "j"], default="js", help="SDK 关节指令：js（默认，同 NERO 工程）或 j")
    ap.add_argument("--gripper-force", type=float, default=2.0, help="夹爪力（N），NERO 工程现场值 2.0")


def open_lab(args, need_camera: bool = True) -> LabDevices:
    from ..hardware.nero_control import NeroController
    dc1 = grabber = None
    if need_camera:
        from ..hardware.dc1_camera import DC1Camera, DC1Config
        dc1 = DC1Camera(DC1Config(serial=args.serial)).start()
        grabber = FrameGrabber(dc1).start()
    arm = NeroController(args.can, args.firmware, allow_motion=args.execute, joint_mode=args.joint_mode,
                         gripper_force_n=args.gripper_force).connect()
    return LabDevices(arm, grabber, RealClock(), dc1)


def safe_belt_z(cfg, config_path) -> float:
    """The higher of the configured belt height and the one calibration step5 measured with the wrist camera."""
    meas = measured_belt_z(config_path)
    return max(cfg.belt_surface_z, meas) if meas is not None else cfg.belt_surface_z


def preflight(dev: LabDevices, cfg, need_start_pose: bool = True) -> list[str]:
    """Read-only checks before anything moves; prints a ✓/✗ line per item and returns the failures."""
    fails = []

    def item(ok: bool, text: str, hint: str = ""):
        print(f"  {'✓' if ok else '✗'} {text}" + ("" if ok or not hint else f"\n      → {hint}"))
        if not ok:
            fails.append(text)

    faults = dev.arm.faults()
    item(not faults, "机械臂无故障" + ("" if not faults else "：" + "；".join(faults)),
         "在厂家上位机 / 示教器上处理（使能、清错）；本程序不会自动使能")
    if dev.camera is not None:
        t_end = time.monotonic() + 3.0
        while dev.camera.latest() is None and time.monotonic() < t_end:
            time.sleep(0.05)
        item(dev.camera.latest() is not None, "DC1 出帧", "检查 USB；先跑 calibration.step0_check_devices --camera")
    if need_start_pose:
        twin = RobotTwin(cfg)
        err = pose_error(twin, dev.arm.joints())
        twin.close()
        d = err["dp_mm"]
        item(pose_ok(err, cfg), f"起始位姿：TCP 偏差 ({d[0]:+.0f}, {d[1]:+.0f}, {d[2]:+.0f}) mm，倾斜 {err['tilt_deg']:.1f}°，"
             f"偏航 {err['yaw_deg']:+.1f}°", "用 step1_record --check-pose 把机械臂摆回等待位")
    # 【说明】静止判定要 1 s 的关节历史；刚连上 CAN 时历史不够，必然判"不静止"。
    #   开相机的步骤恰好先等了出帧，掩盖了这一点；step5 不开相机，所以这里最多等 3 s 再下结论。
    t_end = time.monotonic() + 3.0
    still = dev.arm.stationary()
    while not still and time.monotonic() < t_end:
        time.sleep(0.05)
        still = dev.arm.stationary()
    item(still, "机械臂静止", "确认已松手、退出拖动模式；关节读数 1 s 内变化要 < 3e-4 rad")
    return fails


def confirm_motion(profile: SafetyProfile, extra: list[str] | None = None) -> bool:
    print("\n" + "=" * 60)
    print(f"即将让机械臂运动。档位：{profile.name}（{profile.title}）")
    print(f"  速度比例 ×{profile.speed_scale}；夹爪{'可以闭合' if profile.allow_close_gripper else '锁定张开'}；"
          f"TCP 不低于带面 {profile.tcp_floor_above_belt_m * 100:.1f} cm；"
          + ("指尖不低于带面 {:.0f} mm".format(profile.finger_clearance_m * 1000) if profile.finger_clearance_m is not None else ""))
    for line in extra or []:
        print("  " + line)
    print("  停止：预览窗口按空格或 q / 终端 Ctrl+C（原地停住）；紧急情况直接拍急停。")
    print("=" * 60)
    return input("手放在急停上，确认现场安全后输入 yes 开始：").strip().lower() == "yes"


# =============================================================================
# 【说明】预览窗口：深度图 + 检出框 + 抓取点 + 阶段文字（step2 回放视频的同款画法）。按空格 / q = 停止本回合。
# =============================================================================
class Preview:
    def __init__(self, K: np.ndarray, camera=None, title: str = "closed loop (space/q = stop)"):
        self.K, self.camera, self.title = K, camera, title

    def __call__(self, ep, row) -> bool:
        from .step2_replay import draw
        frame = ep.camera.latest()
        if frame is None:
            return False
        seen = ep.seen
        lines = [f"{row['phase']}  step {ep.steps}  t={row['t']:.1f}s",
                 f"{'VISIBLE' if row.get('visible') else 'mem age %d' % row.get('mem_age', 0)}  width {row['width'] * 100:.1f} cm"]
        if row.get("fixes"):
            lines.append("fix: " + "; ".join(row["fixes"]))
        if seen is not None and seen.det is not None:
            img = draw(frame.depth, seen.det, seen.grasps, seen.visible, self.K, np.linalg.inv(ep.twin.T_base_cam_cv()), lines)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = depth_preview(frame.depth)
            for k, s in enumerate(lines):
                cv2.putText(img, s, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.title, img)
        return (cv2.waitKey(1) & 0xFF) in (ord(" "), ord("q"))

    def idle(self, wait_ms: int = 30):
        """Keep the window alive while the program waits for terminal input (live depth + a 'paused' note)."""
        frame = self.camera.latest() if self.camera is not None else None
        if frame is not None:
            img = depth_preview(frame.depth)
            cv2.putText(img, "PAUSED - answer in the terminal", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(self.title, img)
        cv2.waitKey(wait_ms)

    def close(self):
        cv2.destroyAllWindows()


# =============================================================================
# 【说明】回合之间程序在终端里等操作者回车。OpenCV 窗口只有调用 waitKey 时才处理事件，
#   直接用 input() 会让窗口几秒后被桌面判为"无响应"（2026-11-24 现场：点了强制退出，整个程序被杀）。
#   ask() 在等输入的同时每 30 ms 刷新一次预览（显示实时深度和 PAUSED），回车后照常读入。
# =============================================================================
def ask(prompt: str, preview: Preview | None = None) -> str:
    """input() that keeps the preview window responsive while waiting."""
    if preview is None:
        return input(prompt)
    print(prompt, end="", flush=True)
    while not select.select([sys.stdin], [], [], 0)[0]:
        preview.idle()
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.rstrip("\n")


def between_episodes(session, q_start: np.ndarray, preview: Preview | None = None) -> bool:
    """Operator menu after an episode; returns False to quit."""
    if not session.arm.can_move:
        return ask("只读预演：回车继续下一回合，q 退出：", preview).strip().lower() != "q"
    while True:
        c = ask("\n机械臂停在原地。回车 = 张开夹爪并回起点（下一回合）｜o = 只张开｜r = 只回起点｜q = 退出：", preview).strip().lower()
        if c == "q":
            return False
        if c in ("", "o"):
            session.set_gripper(0.10)
            session.clock.sleep_until(session.clock.now() + 1.0)
        if c in ("", "r"):
            problems = session.move_joints(q_start, speed_rad_s=0.3)
            if problems:
                print("回起点中途停下：" + "；".join(problems))
                continue
        if c == "":
            return True


# =============================================================================
# 【说明】现场测量程序（step8～10）的公共开头：仿真或真机 → 开机检查 → 执行时输入 yes → 建 LoopSession。
#   --sim：MuJoCo 虚拟实验室（同 step4/step5 --sim），sim_setup(cell) 摆好场景（例如挪走或放好工件）
#   真机：不加 --execute 是只读预演（CAN 发送被禁止，整条链路照跑）。
#   返回的 run.close() 停止发送、关设备；输出目录在 SIM_OUTPUT_DIR/<name>_sim 或 REAL_DATA_DIR 下。
# =============================================================================
@dataclass
class MeasureRun:
    session: object
    arm: object
    camera: object | None
    clock: object
    out: object
    mode: str
    cell: object | None
    _closers: list

    def close(self):
        for f in self._closers:
            f()


def add_measure_args(ap: argparse.ArgumentParser):
    add_lab_args(ap)
    ap.add_argument("--sim", action="store_true", help="在 MuJoCo 虚拟实验室里跑（不连硬件）")
    ap.add_argument("--no-viewer", dest="viewer", action="store_false")
    ap.add_argument("--out-root", default=None, help="报告目录的上级（默认：真机 REAL_DATA_DIR，--sim 为 SIM_OUTPUT_DIR/<名字>_sim）")


def open_measure_run(args, cfg, profile: SafetyProfile, name: str, actions: list[str], need_camera: bool = False,
                     sim_setup=None, seed: int = 0, sim_cfg=None) -> MeasureRun | None:
    from pathlib import Path
    from ... import REAL_DATA_DIR, SIM_OUTPUT_DIR
    from .runtime.session import LoopSession
    root = Path(args.out_root) if args.out_root else (SIM_OUTPUT_DIR / f"{name}_sim" if args.sim else REAL_DATA_DIR)
    out = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    if args.sim:
        from ...envs.viewer import display_available
        from ..hardware.sim_workcell import SimWorkcell
        cell = SimWorkcell(sim_cfg or cfg, viewer=args.viewer and display_available())   # sim_cfg：仿真世界（例如按实测工件尺寸）
        cell.reset(seed=seed, object_types=["box"])
        (sim_setup or (lambda c: c.park_object()))(cell)
        arm, camera, clock, belt_z, closers, mode = cell.arm, cell.camera, cell.clock, cfg.belt_surface_z, [cell.close], "sim"
    else:
        cell = None
        dev = open_lab(args, need_camera=need_camera)
        arm, camera, clock, closers = dev.arm, dev.camera, dev.clock, [dev.close]
        belt_z, mode = safe_belt_z(cfg, args.config), "execute" if args.execute else "rehearsal"
        print("\n开机检查：")
        if preflight(dev, cfg):
            print("\n检查没通过，未启动。")
            dev.close()
            return None
        if args.execute:
            if not confirm_motion(profile, actions):
                print("没有确认，未启动。")
                dev.close()
                return None
            arm.prepare_motion()
    latency = 0.0 if args.sim else cfg.camera.latency_s
    session = LoopSession(cfg, profile, arm, camera, clock, belt_z_safe=belt_z, latency_s=latency)
    return MeasureRun(session, arm, camera, clock, out, mode, cell, [session.close] + closers)
