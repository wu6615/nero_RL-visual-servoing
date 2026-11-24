"""Step 1 - capture eye-in-hand calibration samples (read-only: the operator moves the arm).

  python -m nero_vs_rl.sim2real.calibration.step1_capture_handeye                  # 真机
  python -m nero_vs_rl.sim2real.calibration.step1_capture_handeye --sim --auto 20  # 仿真干跑（无窗口）

窗口按键：空格/c 采一张  u 撤销上一张  q/Esc 结束并保存
"""
from __future__ import annotations

import argparse
import time

import cv2
import imageio.v2 as imageio
import numpy as np

from .board import Chessboard, detect, draw, solve_pnp
from .common import MIN_ROTATION_SPAN_DEG, MIN_SAMPLES, RobotFK, append_jsonl, new_session_dir, write_json
from .handeye import Sample, motion_quality

# =============================================================================
# 【说明】一次采样做了什么（capture）：
#   1. 机械臂必须已静止 ≥ hold 秒（相机和 CAN 没有硬件同步，只能在静止时配对）
#   2. 记下关节角 → 取一帧新图 → 再读关节角，前后不一致就作废（拍照期间动了）
#   3. 检测棋盘格角点 + PnP（出厂内参），PnP 误差大就提示重拍
#   4. 保存：彩色 PNG、深度 npy、关节角、MJCF 正解的 gripper_base 位姿、SDK 法兰位姿、角点
# 姿态建议：15~25 张；每张都要看到整块板；在板上方不同方向倾斜 20°~40°，并绕相机光轴转一转；
#   距离 25~45 cm；不要只做平移（平移不提供手眼标定需要的信息）。
# =============================================================================


class Sampler:
    def __init__(self, out, cam, arm, fk: RobotFK, board: Chessboard, hold_s: float):
        self.out, self.cam, self.arm, self.fk, self.board, self.hold_s = out, cam, arm, fk, board, hold_s
        self.K, self.D = cam.color_intrinsics.K, cam.color_intrinsics.D   # 驱动保证有出厂内参，否则 start() 就报错
        self.samples: list[Sample] = []
        self.records: list[dict] = []
        (out / "images").mkdir()

    def capture(self) -> str:
        if not self.arm.stationary(self.hold_s):
            return f"机械臂还没静止 {self.hold_s:.1f} s，松手等一下再采"
        q0 = self.arm.joints()
        self.cam.read()                              # SDK v1 帧组有队列：先丢一组，保证用的是静止之后拍的
        frame = self.cam.read()
        q1 = self.arm.joints()
        if np.max(np.abs(q1 - q0)) > 3e-4 or not self.arm.stationary(self.hold_s):
            return "拍照期间机械臂动了，作废"
        if frame.color is None:
            return "这一帧没有彩色图"
        corners = detect(frame.color, self.board)
        if corners is None:
            return "没检测到完整的棋盘格（要看到全部 11x8 个内角点，夹爪张开别挡住）"
        T_cb, rms = solve_pnp(corners, self.board, self.K, self.D)
        q = (q0 + q1) / 2
        T_bg = self.fk.body_T(q, "gripper_base")
        flange = self.arm.flange_T()
        sid = (self.records[-1]["id"] + 1) if self.records else 0
        name = f"{sid:03d}"
        imageio.imwrite(self.out / "images" / f"{name}_color.png", frame.color)
        if frame.depth is not None:
            np.save(self.out / "images" / f"{name}_depth.npy", frame.depth.astype(np.float32))
        rec = dict(id=sid, t=time.time(), joints=q, T_base_gripper_base=T_bg,
                   T_base_flange_sdk=flange, corners=corners, pnp_rms_px=rms, T_cam_board_factory=T_cb,
                   image=f"images/{name}_color.png", depth=f"images/{name}_depth.npy" if frame.depth is not None else None)
        self.records.append(rec)
        self.samples.append(Sample(T_bg, T_cb, corners, rms, sid))
        self._rewrite()
        mq = motion_quality(self.samples)
        warn = "  ⚠ PnP 误差偏大，检查对焦/反光" if rms > 1.0 else ""
        return (f"样本 #{sid} OK（共 {len(self.samples)} 张）：PnP {rms:.2f} px，板距 {np.linalg.norm(T_cb[:3, 3]) * 100:.0f} cm，"
                f"累计最大旋转 {mq['max_rel_rot_deg']:.0f}°{warn}")

    def undo(self) -> str:
        if not self.records:
            return "没有可撤销的样本"
        rec = self.records.pop()
        self.samples.pop()
        self._rewrite()
        return f"已撤销样本 #{rec['id']}（剩 {len(self.samples)} 张）"

    def _rewrite(self):
        path = self.out / "samples.jsonl"
        path.write_text("", encoding="utf-8")
        for r in self.records:
            append_jsonl(path, r)

    def status(self) -> str:
        mq = motion_quality(self.samples) if len(self.samples) >= 2 else dict(max_rel_rot_deg=0, axis_diversity=[0, 0, 0])
        return (f"samples {len(self.samples)}/{MIN_SAMPLES}+  rot span {mq['max_rel_rot_deg']:.0f}/{MIN_ROTATION_SPAN_DEG:.0f} deg  "
                f"axis div {min(mq['axis_diversity']):.2f}")


def _overlay(img, lines, ok_flags):
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i, (text, ok) in enumerate(zip(lines, ok_flags)):          # cv2 只能画英文
        cv2.putText(out, text, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
    return out


def run_interactive(sampler: Sampler):
    print("窗口已打开：空格/c 采样，u 撤销，q 结束。用示教/拖动模式摆姿态，松手静止后再按空格。")
    win = "step1 hand-eye capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    board = sampler.board
    while True:
        frame = sampler.cam.read()
        img = frame.color if frame.color is not None else np.zeros((480, 640, 3), np.uint8)
        # 预览只求快：半分辨率，先用 1 ms 的快速检测，失败再用扇区法（~15 ms，更稳）；采样时另做全分辨率亚像素检测
        gray = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (board.cols, board.rows), cv2.CALIB_CB_FAST_CHECK)
        if not found:
            found, corners = cv2.findChessboardCornersSB(gray, (board.cols, board.rows))
        preview = draw(img, corners.reshape(-1, 2) * 2 if found else None, board)
        still = sampler.arm.stationary(sampler.hold_s)
        lines = [f"board: {'OK' if found else 'NOT FOUND'}", f"arm still: {'YES' if still else 'no'}", sampler.status(),
                 "SPACE/c capture  u undo  q quit"]
        cv2.imshow(win, _overlay(preview, lines, [found, still, len(sampler.samples) >= MIN_SAMPLES, True]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), ord("c")):
            print(sampler.capture())
        elif key == ord("u"):
            print(sampler.undo())
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description="手眼标定第 1 步：采样（只读机械臂，不发运动指令）")
    ap.add_argument("--sim", action="store_true", help="用 MuJoCo 仿真相机/机械臂干跑")
    ap.add_argument("--auto", type=int, default=0, help="仅仿真：自动生成 N 个姿态并采样（无窗口）")
    ap.add_argument("--cols", type=int, default=11, help="棋盘格内角点列数")
    ap.add_argument("--rows", type=int, default=8, help="棋盘格内角点行数")
    ap.add_argument("--square", type=float, default=0.02, help="方格边长（米）")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--firmware", default="v121")
    ap.add_argument("--serial", default=None, help="相机序列号（多台相机时用）")
    ap.add_argument("--hold", type=float, default=1.0, help="判定静止所需时间（秒）")
    ap.add_argument("--out", default=None, help="输出目录（默认 calib_data/<时间>_handeye）")
    args = ap.parse_args(argv)
    if args.auto and not args.sim:
        ap.error("--auto 只能和 --sim 一起用：真机姿态由操作者手动摆，本程序不驱动机械臂")

    board = Chessboard(args.cols, args.rows, args.square)
    fk = RobotFK()
    if args.sim:
        from ..hardware.sim_devices import SimDC1Camera, SimNeroArm, SimRig
        rig = SimRig(board)
        cam, arm = SimDC1Camera(rig), SimNeroArm(rig)
    else:
        from ..hardware.dc1_camera import DC1Camera, DC1Config
        from ..hardware.nero_arm import NeroArmReader
        cam, arm = DC1Camera(DC1Config(serial=args.serial)), NeroArmReader(args.can, args.firmware)

    from pathlib import Path
    out = Path(args.out) if args.out else new_session_dir("handeye_sim" if args.sim else "handeye")
    out.mkdir(parents=True, exist_ok=True)
    with cam, arm:
        info = cam.camera_info()
        write_json(out / "camera_info.json", info)
        for w in info["warnings"]:
            print("⚠", w)
        write_json(out / "session.json", dict(kind="eye_in_hand", board=board.to_dict(), sim=args.sim,
                                              robot_pose="MJCF FK of measured joints -> gripper_base",
                                              can=args.can, firmware=args.firmware, created=time.strftime("%F %T")))
        sampler = Sampler(out, cam, arm, fk, board, args.hold)
        if args.auto:
            from ...envs.dynamic_grasp import DynamicGraspEnv
            q_seed = DynamicGraspEnv().q_home
            for q in rig.random_view_joints(q_seed, args.auto):
                arm.move_to(q)
                print(sampler.capture())
        else:
            run_interactive(sampler)
    print(f"\n采样结束：{len(sampler.samples)} 张，数据在 {out}")
    print(f"下一步：python -m nero_vs_rl.sim2real.calibration.step2_solve_handeye --session {out}")
    return out


if __name__ == "__main__":
    main()
