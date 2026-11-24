"""Step 1 (offline stage) - record DC1 depth/colour + NERO joint angles while objects pass on the belt.

  python -m nero_vs_rl.sim2real.deploy.step1_record --seconds 30                # 真机：只读，机械臂不动
  python -m nero_vs_rl.sim2real.deploy.step1_record --sim --seconds 20          # 仿真干跑（带物体真值）
  python -m nero_vs_rl.sim2real.deploy.step1_record --check-pose                # 摆等待位：实时显示离仿真初始位姿多远

机械臂不会被使能、不会收到任何运动指令；操作者先用示教/拖动把它摆到等待位（相机朝下看带），
需要时也可以在录制中用拖动模式手动跟着物体走（检验盲区和记忆外推）。按 q 或 Ctrl+C 结束。
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ... import REAL_DATA_DIR, TRAIN_CONFIG
from ...utils.config import load_config
from .common import RecordWriter, new_record_dir

# =============================================================================
# 【说明】为什么先录再离线跑：
#   真机上第一次让策略闭环控制之前，要先确认"策略看到的东西"和仿真里一样——
#   物体检测得准不准、速度估计对不对、盲区什么时候出现、策略想往哪里动。
#   这些都能用录下来的数据在电脑上反复离线检查（step2_replay），全程机械臂不动，没有风险。
# 频率：按控制频率（默认 10 Hz）取帧，与策略的控制周期一致；相机约 24 帧/秒（USB2.0），够用。
# 关节角按相机帧到达时间在 100 Hz 关节历史里插值（相机和 CAN 没有硬件同步）。
# =============================================================================


def depth_preview(depth: np.ndarray, max_m: float = 0.8) -> np.ndarray:
    v = np.clip(depth / max_m, 0, 1)
    img = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth <= 0] = 0
    return img


def pose_error(twin, q) -> dict:
    """TCP of joints ``q`` vs the sim's start pose: position offset, tilt of the approach axis, yaw about vertical."""
    p_home, R_home = twin.home_tcp()
    twin.set_joints(q)
    p, R = twin.tcp_pose()
    tilt = float(np.degrees(np.arccos(np.clip(R[:, 2] @ R_home[:, 2], -1, 1))))
    yaw = float(np.degrees(np.arctan2(R[1, 1], R[0, 1]) - np.arctan2(R_home[1, 1], R_home[0, 1])))
    yaw = (yaw + 180) % 360 - 180
    return dict(dp_mm=(p - p_home) * 1000, tilt_deg=tilt, yaw_deg=yaw)


# 【说明】训练时初始位姿 = 仿真等待位 + 每轴 ±init_pos_noise_m 平移 + 绕竖直轴 ±init_yaw_noise_rad 偏航，
#   没有倾斜。真机起点落在这个范围内，策略才处在熟悉的状态。
def pose_ok(err: dict, cfg) -> bool:
    return bool(np.all(np.abs(err["dp_mm"]) <= cfg.init_pos_noise_m * 1000) and err["tilt_deg"] <= 5.0
                and abs(err["yaw_deg"]) <= np.degrees(cfg.init_yaw_noise_rad))


def check_pose(args, cfg):
    """Read-only live readout while the operator drags the arm to the standby pose."""
    from ..hardware.nero_arm import NeroArmReader
    from .common import RobotTwin

    twin = RobotTwin(cfg)
    print("仿真初始位姿对应的关节角（rad，供参考）:", np.round(twin.env.q_home, 4).tolist())
    print(f"合格范围：每轴 ±{cfg.init_pos_noise_m * 1000:.0f} mm，倾斜 ≤ 5°，偏航 ±{np.degrees(cfg.init_yaw_noise_rad):.0f}°。"
          "Ctrl+C 结束")
    err = None
    t_end = time.monotonic() + args.seconds
    with NeroArmReader(args.can, args.firmware) as arm:
        try:
            while time.monotonic() < t_end:
                err = pose_error(twin, arm.joints())
                d = err["dp_mm"]
                print(f"\rTCP 偏差 x {d[0]:+6.0f}  y {d[1]:+6.0f}  z {d[2]:+6.0f} mm   倾斜 {err['tilt_deg']:4.1f}°   "
                      f"偏航 {err['yaw_deg']:+5.1f}°   {'✓ 可以录' if pose_ok(err, cfg) else '✗ 继续调'}   ", end="", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    print()
    twin.close()
    return err


def record_real(args, cfg, out):
    from ..hardware.dc1_camera import DC1Camera, DC1Config
    from ..hardware.nero_arm import NeroArmReader

    period = 1.0 / args.hz
    with DC1Camera(DC1Config(serial=args.serial)) as cam, NeroArmReader(args.can, args.firmware) as arm:
        info = cam.camera_info()
        meta = dict(source="real", hz=args.hz, config=str(args.config), camera=info, note=args.note,
                    created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    joints_note="frames 里的关节角取在帧到达时刻（未补偿相机延迟）；joints_hist.npz 是完整 100 Hz 关节历史")
        w = RecordWriter(out, meta)
        print(f"录制到 {out}（{args.seconds:.0f} s，{args.hz:g} Hz）。按 q 或 Ctrl+C 结束")
        t_end, next_t = time.monotonic() + args.seconds, None
        hist, last_t = [], -np.inf                                 # 完整 100 Hz 关节历史（回放时按相机延迟重新取关节）
        try:
            while time.monotonic() < t_end:
                fr = cam.read()
                # 按 10 Hz 节拍取帧：节拍到了（容差 20 ms，约半个相机帧）才收；落后太多就从当前帧重新起拍
                if next_t is not None and fr.stamp < next_t - 0.02:
                    continue
                next_t = fr.stamp + period if next_t is None or fr.stamp > next_t + period else next_t + period
                q = arm.joints_at(fr.stamp)
                w.write(fr.stamp, q, fr.depth, fr.color, episode=0)
                new = arm.history(last_t)
                if new:
                    hist += new
                    last_t = new[-1][0]
                if not args.no_preview:
                    img = depth_preview(fr.depth)
                    cv2.putText(img, f"REC {w.n} frames  q to stop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.imshow("step1_record (depth)", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            w.close()
            cv2.destroyAllWindows()
            if hist:
                np.savez_compressed(out / "joints_hist.npz", t=np.array([h[0] for h in hist]),
                                    q=np.array([h[1] for h in hist]))
    return w.n


def record_sim(args, cfg, out):
    """Dry run: the env's wrist camera stands in for the DC1; ground truth is stored per frame."""
    from ...algorithms.scripted_pbvs import ScriptedGraspPolicy
    from ...envs.dynamic_grasp import DynamicGraspEnv

    env = DynamicGraspEnv(cfg)
    c = cfg.camera
    fy = c.intrinsics[1] if c.intrinsics is not None else (c.height / 2) / np.tan(np.deg2rad(c.fovy_deg) / 2)
    # MuJoCo 渲染是理想针孔：主点在画面中心、无畸变（与真机 DC1 出厂内参略有不同，录在 meta 里）
    intr = dict(width=c.width, height=c.height, fx=fy, fy=fy, cx=c.width / 2, cy=c.height / 2, dist=[0.0] * 5,
                model="sim_pinhole")
    meta = dict(source="sim", hz=1.0 / env.control_dt, config=str(args.config), note=args.note,
                camera=dict(depth_intrinsics=intr, warnings=["仿真相机（干跑），不是真机数据"]),
                sim_motion=args.sim_motion, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    w = RecordWriter(out, meta)
    policy = ScriptedGraspPolicy(cfg) if args.sim_motion == "scripted" else None
    hold = np.r_[np.zeros(6), 1.0]                              # 不动、夹爪全开（对应真机"机械臂不动"）
    n_frames, ep = int(args.seconds * args.hz), 0
    obs, info = env.reset(seed=args.seed)
    try:
        for _ in range(n_frames):
            rgb, depth = env.render_wrist_rgbd()
            gs = env.model.geom_size[env.obj["geom"]]
            half = gs.copy() if env.obj_type == "box" else np.array([gs[0], gs[0], gs[1]])
            w.write(env.t, env.agent.arm_qpos(env.data), depth, rgb, episode=ep,
                    truth=dict(obj_type=env.obj_type, T_wo=env.obj_T(), half=half))
            a = policy(obs, info) if policy is not None else hold
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                ep += 1
                obs, info = env.reset(seed=args.seed + ep)
                if policy is not None:
                    policy.reset()
    finally:
        w.close()
        env.close()
    return w.n


def main(argv=None):
    ap = argparse.ArgumentParser(description="离线阶段：录制 DC1 深度 + NERO 关节（机械臂不动）")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=10.0, help="取帧频率，默认等于控制频率")
    ap.add_argument("--config", default=str(TRAIN_CONFIG))
    ap.add_argument("--out", default=None, help=f"录制目录（默认 {REAL_DATA_DIR}/<时间>_record）")
    ap.add_argument("--note", default="", help="备注：物体是什么、带速多少等，会写进 meta.json")
    ap.add_argument("--sim", action="store_true", help="仿真干跑：用仿真腕部相机代替 DC1")
    ap.add_argument("--sim-motion", choices=["hold", "scripted"], default="hold",
                    help="仿真里机械臂：hold 不动（对应真机）；scripted 用脚本基线去抓（检验盲区阶段）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--firmware", default="v121")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--check-pose", action="store_true", help="只读：实时显示 TCP 离仿真初始位姿多远（摆等待位用），不录制")
    args = ap.parse_args(argv)
    cfg, _ = load_config(args.config)
    if args.check_pose:
        if "--seconds" not in (argv or __import__("sys").argv):
            args.seconds = 1e9                                  # 摆位检查默认一直运行，Ctrl+C 结束
        return check_pose(args, cfg)
    if args.sim and args.hz != 1.0 * cfg.control_freq:
        args.hz = cfg.control_freq                              # 仿真按控制步取帧
    if args.out:
        from pathlib import Path
        out = Path(args.out)
        (out / "depth").mkdir(parents=True, exist_ok=False)
        (out / "color").mkdir()
    else:
        out = new_record_dir("sim_record" if args.sim else "record")
    n = record_sim(args, cfg, out) if args.sim else record_real(args, cfg, out)
    print(f"已录 {n} 帧：{out}")
    print(f"回放这一段（可直接复制）：\npython -m nero_vs_rl.sim2real.deploy.step2_replay --record {out} --video")
    return out


if __name__ == "__main__":
    main()
