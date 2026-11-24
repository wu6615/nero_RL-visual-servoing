"""NERO7 + AGX gripper with motion commands: the only class in this project that can make the arm move.

Counterpart of GAP-RL ``sim2real/robot_control.py`` + ``gripper_control.py``.  SDK usage (gripper effector,
fault fields, ``move_js`` streaming, damped emergency stop) follows the NERO project's field-tested
``runtime/nero_pick_hardware.py`` (read-only reference; nothing is imported from it).
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .nero_arm import NeroArmReader

# =============================================================================
# 【说明】安全约定（和 NERO 工程一致）：
#   · 不使能、不复位、不改 CAN：使能由操作者在厂家上位机 / 示教器上做。没使能就报故障退出，不会去"帮忙"使能。
#   · allow_motion=False（step 脚本没加 --execute）= 只读预演：连接前把 SDK 的 CAN 发送函数换成"直接报错"，
#     之后无论哪段代码误调用了运动接口，一个字节也发不出去。
#   · 关节指令只经 send_joints 发出，由 runtime/streamer.py 以 100 Hz、单点 ≤ 档位限速 × 10 ms 的插值点调用。
#     用 move_js（SDK 注释"无平滑、尽快到达"）是因为插值已经在本机做了；NERO 工程现场也是 js + 100 Hz。
#     --joint-mode j 可换成 move_j（控制器自带规划，跟随会更慢）。
#   · 急停 emergency_stop = SDK electronic_emergency_stop（阻尼停止），只在机械臂报故障或发送出错时用；
#     普通的"停"是发送器原地保持（见 runtime/episode.py），不调用急停。
#   · 所有发送都加锁：100 Hz 发送线程和主线程（夹爪、急停）不会同时写 CAN。
#
# faults() 检查哪些反馈（字段名同 NERO 工程）：
#   CAN 通信错误；机械臂状态 arm_status / err_code ≠ 0；控制模式 ctrl_mode ∉ {0,1,3}（示教或别的程序在控制）；
#   示教状态 teach_status ∉ {0,2,6}；7 个关节驱动：未使能、欠压、过热、过流、碰撞、堵转、驱动错误；
#   夹爪：不是宽度模式、未使能、驱动报错；关节/状态反馈超过 0.2 s 没更新。
# =============================================================================
FEEDBACK_MAX_AGE_S = 0.2
DRIVER_FLAGS = {"voltage_too_low": "欠压", "motor_overheating": "电机过热", "driver_overcurrent": "过流",
                "driver_overheating": "驱动过热", "collision_status": "碰撞", "stall_status": "堵转",
                "driver_error_status": "驱动错误"}
GRIPPER_FLAGS = {"voltage_too_low": "欠压", "motor_overheating": "电机过热", "driver_overcurrent": "过流",
                 "driver_overheating": "驱动过热", "sensor_status": "传感器", "driver_error_status": "驱动错误"}


class NeroController(NeroArmReader):
    def __init__(self, channel: str = "can0", firmware: str = "v121", allow_motion: bool = False,
                 joint_mode: str = "js", gripper_force_n: float = 2.0):
        super().__init__(channel, firmware)
        if joint_mode not in ("js", "j"):
            raise ValueError("joint_mode 只能是 js 或 j")
        self.can_move = bool(allow_motion)
        self.joint_mode, self.gripper_force_n = joint_mode, gripper_force_n
        self.gripper = None
        self._send_lock = threading.Lock()

    # ------------------------------------------------------------------ 连接
    def connect(self, timeout_s: float = 3.0) -> "NeroController":
        # 【说明】父类只等到关节角就返回；机械臂状态、7 个驱动、夹爪反馈是各自的 CAN 帧，
        #   刚连上时常有几项还没到（2026-11-24 真机实测：8 次里 5 次缺，5～20 ms 内补齐）。
        #   不等的话，开机检查会把"还没收到"误报成故障。这里最多再等 1 s；真的一直没有，交给 faults() 报。
        super().connect(timeout_s)
        t_end = time.monotonic() + 1.0
        while any(f.startswith("没有") for f in self.faults()) and time.monotonic() < t_end:
            time.sleep(0.005)
        return self

    def _before_connect(self):
        self.gripper = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        comm = self.robot.create_comm()                  # connect() 会沿用这个通信对象，夹爪也共用它
        if not self.can_move:
            def forbidden(*_a, **_k):
                raise RuntimeError("只读预演：禁止任何 CAN 发送（运行时没有加 --execute）")
            comm.send = forbidden

    def prepare_motion(self):
        """Before the first motion command: no faults (incl. all joints enabled), speed percent 100 (as the NERO project)."""
        if not self.can_move:
            raise RuntimeError("只读预演不能准备运动")
        problems = self.faults()
        if problems:
            raise RuntimeError("机械臂没准备好：" + "；".join(problems))
        with self._send_lock:
            self.robot.set_speed_percent(100)

    # ------------------------------------------------------------------ 反馈
    def gripper_width(self) -> float:
        msg = self.gripper.get_gripper_status()
        if msg is None:
            raise RuntimeError("还没有夹爪反馈")
        return float(msg.msg.value)

    def faults(self) -> list[str]:
        r, out, now = self.robot, [], time.time()           # SDK 消息时间戳是 time.time()（NERO 工程同样用法）
        if r.has_comm_error():
            return [f"CAN 通信异常：{r.get_comm_error()}"]
        for label, msg in (("关节角", r.get_joint_angles()), ("机械臂状态", r.get_arm_status())):
            if msg is None:
                out.append(f"没有{label}反馈")
            elif now - float(msg.timestamp) > FEEDBACK_MAX_AGE_S:
                out.append(f"{label}反馈 {now - float(msg.timestamp):.2f} s 没更新")
        st = r.get_arm_status()
        if st is not None:
            s = st.msg
            if int(s.arm_status) != 0 or int(s.err_code) != 0:
                out.append(f"机械臂报错：arm_status={s.arm_status}，err_code={s.err_code}")
            if int(s.ctrl_mode) not in (0, 1, 3):
                out.append(f"控制模式 {s.ctrl_mode}：请退出示教、关掉其他控制程序")
            if int(s.teach_status) not in (0, 2, 6):
                out.append(f"示教/轨迹仍在运行（teach_status={s.teach_status}）")
        for j in range(1, 8):
            drv = r.get_driver_states(j)
            if drv is None:
                out.append(f"没有关节 {j} 驱动反馈")
                continue
            foc = drv.msg.foc_status
            if self.can_move and not foc.driver_enable_status:
                out.append(f"关节 {j} 未使能（请操作者在厂家上位机使能；本程序不会自动使能）")
            out += [f"关节 {j} 驱动{name}" for key, name in DRIVER_FLAGS.items() if getattr(foc, key, False)]
        g = self.gripper.get_gripper_status()
        if g is None:
            out.append("没有夹爪反馈")
        else:
            gm = g.msg
            if self.can_move and (gm.mode != "width" or not gm.foc_status.driver_enable_status):
                out.append("夹爪需要已使能、宽度（width）模式")
            out += [f"夹爪{name}" for key, name in GRIPPER_FLAGS.items() if getattr(gm.foc_status, key, False)]
        return out

    # ------------------------------------------------------------------ 指令
    def send_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("关节指令必须是 7 个有限数")
        with self._send_lock:
            getattr(self.robot, "move_" + self.joint_mode)(q.tolist())

    def send_gripper(self, width_m: float) -> None:
        with self._send_lock:
            self.gripper.move_gripper_m(value=float(np.clip(width_m, 0.0, 0.10)), force=self.gripper_force_n)

    def emergency_stop(self) -> None:
        if not self.can_move:
            return
        with self._send_lock:
            self.robot.electronic_emergency_stop()
