"""What the closed loop needs from the arm, the camera and the clock; the lab hardware and the MuJoCo cell both provide it.

Counterpart of the device handles GAP-RL ``sim2real_traj_following.py`` builds at start-up
(``RobotUR`` / ``GripperController`` / ``RealsenseCamera``), written down as explicit interfaces so the
same loop can run against ``hardware/nero_control.py`` (lab) or ``hardware/sim_workcell.py`` (dry run).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

import numpy as np

# =============================================================================
# 【说明】闭环需要的三样东西：
#   ArmDevice     机械臂 + 夹爪：读关节（含按时间插值）、读夹爪开口、查故障、发关节目标（100 Hz）、发夹爪目标、急停
#   CameraDevice  腕部深度相机：最新一帧 + 深度内参
#   Clock         时间：now / sleep_until / every（周期回调）
#                 真机：RealClock，every 开一个线程按绝对节拍调用（100 Hz 发送用）
#                 仿真：SimClock（hardware/sim_workcell.py），sleep_until 推进物理，every 的回调在推进时按节拍调用
#                 ⇒ 同一份闭环代码在仿真里是"确定性、比实时快"的，在真机上是多线程实时的
#
# 夹爪开口的换算（仿真 ↔ 真机）：
#   仿真 gripper_joint1 行程 q ∈ [0, 0.05]，两指对称 ⇒ 总行程 2q ∈ [0, 0.10]；
#   真机 AGX 夹爪"宽度"读数/指令 ∈ [0, 0.10] m（NERO 工程 open_width_m = 0.1）。
#   两者按 width = 2q 对应（仿真指垫间距还要再加 8 mm）。★ 待现场核对：夹一个已知宽度的块，看读数。
# =============================================================================
GRIPPER_WIDTH_PER_Q = 2.0
GRIPPER_MAX_WIDTH_M = 0.10


def q_to_width(q: float) -> float:
    return float(np.clip(GRIPPER_WIDTH_PER_Q * q, 0.0, GRIPPER_MAX_WIDTH_M))


def width_to_q(width_m: float) -> float:
    return float(np.clip(width_m, 0.0, GRIPPER_MAX_WIDTH_M) / GRIPPER_WIDTH_PER_Q)


class ArmDevice(Protocol):
    can_move: bool                      # False = 只读预演：send_* 不会真的发出去

    def joints(self) -> np.ndarray: ...                 # 最新实测关节角（7）
    def joints_at(self, t: float) -> np.ndarray: ...    # 时刻 t（monotonic / 仿真时钟）的关节角，按历史插值
    def gripper_width(self) -> float: ...               # 实测夹爪开口（m）
    def faults(self) -> list[str]: ...                  # 故障（空表 = 正常）：未使能、驱动报错、通信中断、反馈过期
    def send_joints(self, q: np.ndarray) -> None: ...   # 一个 100 Hz 关节位置目标
    def send_gripper(self, width_m: float) -> None: ...
    def emergency_stop(self) -> None: ...


class CameraDevice(Protocol):
    K: np.ndarray                       # 深度相机内参 3×3
    D: np.ndarray                       # 畸变（5）

    def latest(self): ...               # 最新一帧 RGBDFrame（depth m, color, stamp）；还没有帧时 None


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep_until(self, t: float) -> None: ...
    def every(self, period_s: float, fn: Callable[[float], None]) -> None: ...
    def stop(self) -> None: ...


# =============================================================================
# 【说明】真机时钟：time.monotonic。every() 开一个线程，按"起点 + k × 周期"的绝对节拍调用 fn(now)：
#   某一拍晚了就直接跳到下一个还没过去的节拍，不补发（补发会让关节指令瞬间连跳几格）。
# =============================================================================
class RealClock:
    def __init__(self):
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, t: float) -> None:
        dt = t - time.monotonic()
        if dt > 0:
            time.sleep(dt)

    def every(self, period_s: float, fn: Callable[[float], None]) -> None:
        def loop():
            origin, k = time.monotonic(), 0
            while not self._stop.is_set():
                if self._stop.wait(max(0.0, origin + k * period_s - time.monotonic())):
                    break
                fn(time.monotonic())
                k = max(k + 1, int((time.monotonic() - origin) / period_s) + 1)
        th = threading.Thread(target=loop, name=f"every-{period_s * 1000:.0f}ms", daemon=True)
        th.start()
        self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)


# =============================================================================
# 【说明】DC1 取帧线程：DC1Camera.read() 是阻塞的（等下一帧，约 24 帧/秒），控制循环不能等它。
#   后台线程一直读，只保留最新一帧；控制循环每拍取一次 latest()，用帧自带的到达时间戳去找关节角。
# =============================================================================
class FrameGrabber:
    def __init__(self, camera):
        self.camera = camera
        intr = camera.depth_intrinsics
        self.K, self.D = intr.K, intr.D
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="dc1-grabber", daemon=True)

    def start(self) -> "FrameGrabber":
        self._thread.start()
        return self

    def _run(self):
        try:
            while not self._stop.is_set():
                fr = self.camera.read()
                with self._lock:
                    self._frame = fr
        except Exception as e:                     # 相机掉线：控制循环会看到帧过期并停下
            self.error = e

    def latest(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
