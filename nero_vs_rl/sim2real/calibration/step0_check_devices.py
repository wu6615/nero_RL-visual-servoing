"""Step 0 - check the DC1 and the NERO connection before calibrating (read-only, no motion).

  python -m nero_vs_rl.sim2real.calibration.step0_check_devices            # 相机 + 机械臂
  python -m nero_vs_rl.sim2real.calibration.step0_check_devices --camera   # 只查相机
"""
from __future__ import annotations

import argparse
import time

import cv2
import imageio.v2 as imageio
import numpy as np

from .common import RobotFK, new_session_dir, write_json
from .handeye import inv, rot_deg

UDEV_HINT = ("相机没列出来时：\n"
             "  1. 换 USB 口/线；`lsusb | grep 2bc5` 应看到 0657（深度）和 0557（彩色）\n"
             "  2. 安装 udev 规则（操作者执行，需要 sudo）：\n"
             "     sudo cp vendor/orbbec_sdk_v1/OrbbecSDK_v1.10.37/Script/99-obsensor-libusb.rules /etc/udev/rules.d/\n"
             "     sudo udevadm control --reload-rules && sudo udevadm trigger，然后拔插相机\n"
             "  3. 确认 Orbbec SDK v1 在 vendor/orbbec_sdk_v1/OrbbecSDK_v1.10.37/（见 calibration/README.md）")


def check_camera(out, serial):
    from ..hardware.dc1_camera import DC1Camera, DC1Config, list_devices
    devs = list_devices()
    print("Orbbec SDK v1 看到的 DC1：", devs or "无")
    if not devs:
        print(UDEV_HINT)
        return False
    cam = DC1Camera(DC1Config(serial=serial))
    with cam:
        for sensor, items in cam.list_profiles().items():
            print(f"  {sensor}: {len(items)} 种 → {items[:8]}{' ...' if len(items) > 8 else ''}")
        t0, n = time.monotonic(), 0
        for _ in range(30):
            f = cam.read(need_color=False)
            n += 1
        fps = n / (time.monotonic() - t0)
        f = cam.read(need_color="color" in cam.active_profiles)
        info = cam.camera_info()
        write_json(out / "camera_info.json", info)
        if f.color is not None:
            imageio.imwrite(out / "color.png", f.color)
        d = f.depth
        valid = d[d > 0]
        vis = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255 / 1.5), cv2.COLORMAP_JET)
        vis[d <= 0] = 0
        cv2.imwrite(str(out / "depth.png"), vis)
    print(f"  设备 {info['device']}")
    print(f"  使用的流 {info['profiles']}，深度约 {fps:.1f} 帧/s")
    print(f"  深度有效像素 {valid.size / d.size * 100:.0f}%，范围 {valid.min() if valid.size else 0:.3f}~{valid.max() if valid.size else 0:.3f} m")
    for k in ("color_intrinsics", "depth_intrinsics"):
        v = info[k]
        print(f"  {k}: " + (f"{v['width']}x{v['height']} fx={v['fx']:.1f} fy={v['fy']:.1f} cx={v['cx']:.1f} cy={v['cy']:.1f}" if v else "无"))
    print("  深度→彩色出厂外参:", "有" if info["T_color_depth"] else "无")
    for w in info["warnings"]:
        print("  ⚠", w)
    return info["color_intrinsics"] is not None or "color" in info["profiles"]


def check_arm(out, channel, firmware):
    from ..hardware.nero_arm import NeroArmReader
    fk = RobotFK()
    with NeroArmReader(channel, firmware) as arm:
        print(f"CAN: {arm.can_info}")
        time.sleep(1.0)
        q = arm.joints()
        F = arm.flange_T()
        still = arm.stationary(0.8)
    print("  关节角 (deg):", np.round(np.degrees(q), 2).tolist(), "| 静止" if still else "| 在动")
    rec = dict(joints=q, flange_sdk=F)
    if F is not None:
        print("  SDK 法兰位置 (m):", np.round(F[:3, 3], 4).tolist())
        print("  与本工程 MJCF 正解比较（SDK 法兰 → 各 body 的偏移）：")
        for body in ("link7", "gripper_flange", "gripper_base"):
            O = inv(F) @ fk.body_T(q, body)
            rec[f"offset_to_{body}"] = O
            print(f"    {body:15s} 平移 {np.linalg.norm(O[:3, 3]) * 1000:7.1f} mm  旋转 {rot_deg(O[:3, :3]):6.2f}°")
        print("  其中一个应接近 0（或是稳定的小偏移）；多个姿态下都一样才说明两套运动学一致（step2 会再查一次）")
    write_json(out / "arm_check.json", rec)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="标定前检查相机和机械臂连接（只读）")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--can", default="can0")
    ap.add_argument("--firmware", default="v121")
    args = ap.parse_args(argv)
    both = not (args.camera or args.arm)
    out = new_session_dir("check")
    ok = True
    if args.camera or both:
        print("== 相机 ==")
        try:
            ok &= bool(check_camera(out, args.serial))
        except Exception as e:
            ok = False
            print("  ❌", e)
            if "没找到" in str(e):                   # 找到了相机但打不开（如出厂标定不匹配）时，udev 提示没用
                print(UDEV_HINT)
    if args.arm or both:
        print("== 机械臂（只读，不使能）==")
        try:
            ok &= check_arm(out, args.can, args.firmware)
        except Exception as e:
            ok = False
            print("  ❌", e)
    print(f"\n{'✅ 可以开始标定' if ok else '❌ 先解决上面的问题'}；记录在 {out}")


if __name__ == "__main__":
    main()
