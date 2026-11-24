"""Read-only NERO7 feedback over CAN via pyAgxArm: joint angles, SDK flange pose, stationarity.

Never enables the arm, never sends motion commands, never changes CAN state.
The operator moves the arm (teach / drag mode or the vendor app); this class only reads.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque

import numpy as np

# =============================================================================
# 【说明】标定用哪个位姿？
#   主用：关节角 → 本工程 MJCF 正解 → gripper_base 位姿（与仿真同一套运动学，标定结果可直接放进仿真）。
#   旁证：SDK 的 get_flange_pose()（[x,y,z,roll,pitch,yaw]，静态 XYZ 欧拉：R = Rz·Ry·Rx）。
#         两者之间应是一个固定变换；step2 会检查它在所有样本上是否恒定，以核对基座系是否一致。
# 静止判定：100 Hz 读关节，最近 hold_s 秒内各关节变化都 < tol_rad 才算静止。
# 相机与 CAN 没有硬件同步，所以只在机械臂静止时采样（同 NERO 工程 teach_recording 的做法）。
# =============================================================================


def euler_xyz_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def flange_pose_to_T(pose6) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = euler_xyz_to_R(*pose6[3:6])
    T[:3, 3] = pose6[:3]
    return T


def check_can_interface(channel: str, expected_bitrate: int = 1_000_000) -> dict:
    """Inspect SocketCAN read-only (ip -details -json link show); raises with a Chinese hint."""
    try:
        out = subprocess.run(["ip", "-details", "-json", "link", "show", "dev", channel],
                             capture_output=True, text=True, timeout=3)
    except FileNotFoundError as e:
        raise RuntimeError("缺少 ip 命令（iproute2）") from e
    if out.returncode:
        raise RuntimeError(f"找不到 CAN 接口 {channel}：检查 USB-CAN 是否插好。{out.stderr.strip()}")
    info = json.loads(out.stdout)[0]
    state = info.get("operstate")
    bitrate = info.get("linkinfo", {}).get("info_data", {}).get("bittiming", {}).get("bitrate")
    if state not in ("UP", "UNKNOWN") or "UP" not in info.get("flags", []):
        raise RuntimeError(f"{channel} 没有启动（state={state}）。请操作者执行：\n"
                           f"  sudo ip link set dev {channel} type can bitrate {expected_bitrate} && "
                           f"sudo ip link set dev {channel} up")
    if bitrate is not None and int(bitrate) != expected_bitrate:
        raise RuntimeError(f"{channel} 波特率 {bitrate}，应为 {expected_bitrate}")
    return dict(channel=channel, state=state, bitrate=bitrate)


def interp_joints(hist: list[tuple[float, np.ndarray]], t: float) -> np.ndarray:
    ts = np.array([h[0] for h in hist])
    if t <= ts[0]:
        return hist[0][1].copy()
    if t >= ts[-1]:
        return hist[-1][1].copy()
    k = int(np.searchsorted(ts, t))
    (t0, q0), (t1, q1) = hist[k - 1], hist[k]
    w = (t - t0) / max(t1 - t0, 1e-9)
    return (1 - w) * q0 + w * q1


class NeroArmReader:
    def __init__(self, channel: str = "can0", firmware: str = "v121", poll_hz: float = 100.0):
        self.channel, self.firmware, self.poll_hz = channel, firmware, poll_hz
        self.robot = None
        self._hist: deque = deque(maxlen=int(poll_hz * 5))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def connect(self, timeout_s: float = 3.0) -> "NeroArmReader":
        self.can_info = check_can_interface(self.channel)
        self.robot = self._create_robot()
        self._before_connect()
        self.robot.connect()                         # 只开始接收反馈；不使能、不发运动指令
        t0 = time.monotonic()
        while self._read_joints() is None:
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError("连上 CAN 但 3 s 内没收到关节反馈：检查机械臂电源、固件版本参数 --firmware")
            time.sleep(0.01)
        self._thread = threading.Thread(target=self._poll, name="nero-joint-poll", daemon=True)
        self._thread.start()
        return self

    def _create_robot(self):
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config
        except ImportError as e:
            raise RuntimeError("没有 pyAgxArm：见 requirements-hardware.txt") from e
        cfg = create_agx_arm_config(robot="nero", firmeware_version=self.firmware, interface="socketcan",
                                    channel=self.channel, receive_own_messages=False)
        return AgxArmFactory.create_arm(cfg)

    def _before_connect(self):
        """Hook for subclasses (nero_control.NeroController attaches the gripper and the send guard here)."""

    def _read_joints(self):
        msg = self.robot.get_joint_angles()
        return None if msg is None else np.asarray(msg.msg, dtype=float)

    def _poll(self):
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            q = self._read_joints()
            if q is not None:
                with self._lock:
                    self._hist.append((time.monotonic(), q))
            time.sleep(period)

    # ---- 读数
    def joints(self) -> np.ndarray:
        with self._lock:
            if not self._hist:
                raise RuntimeError("还没有关节反馈")
            return self._hist[-1][1].copy()

    def joints_at(self, t: float) -> np.ndarray:
        """Joint angles at monotonic time ``t`` (linear interpolation in the 100 Hz history).

        相机帧和关节反馈没有硬件同步：按相机帧的到达时间去历史里插值，比直接取"最新一帧关节"更准
        （操作者拖动机械臂时差别明显；静止时两者一样）。
        """
        with self._lock:
            hist = list(self._hist)
        if not hist:
            raise RuntimeError("还没有关节反馈")
        return interp_joints(hist, t)

    def history(self, since: float = -np.inf) -> list[tuple[float, np.ndarray]]:
        """100 Hz (time, joints) samples newer than ``since`` (the buffer keeps the last 5 s)."""
        with self._lock:
            return [(t, q.copy()) for t, q in self._hist if t > since]

    def flange_T(self) -> np.ndarray | None:
        msg = self.robot.get_flange_pose()
        return None if msg is None else flange_pose_to_T(np.asarray(msg.msg, dtype=float))

    def stationary(self, hold_s: float = 1.0, tol_rad: float = 3e-4) -> bool:
        now = time.monotonic()
        with self._lock:
            recent = [q for t, q in self._hist if now - t <= hold_s]
            span = self._hist[-1][0] - self._hist[0][0] if self._hist else 0.0
        if span < hold_s or len(recent) < 0.5 * hold_s * self.poll_hz:
            return False
        recent = np.array(recent)
        return bool(np.all(recent.max(0) - recent.min(0) < tol_rad))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.robot is not None:
            self.robot.disconnect()                  # SDK 文档：只释放读线程和 CAN 资源，不改变机械臂状态

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()
