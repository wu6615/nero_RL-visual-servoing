"""MuJoCo interactive viewer for watching evaluation rollouts (passive viewer, real-time pacing, text overlay)."""
from __future__ import annotations

import os
import time

import mujoco
import numpy as np


# =============================================================================
# 【说明】评估时打开 MuJoCo 图形界面看效果：
#   - 用 passive viewer：物理仍由环境自己步进，viewer 只负责显示；窗口里可以用鼠标转视角、
#     双击选中物体、按 Tab/Shift+Tab 开关左右面板
#   - 环境一个控制步内部有 50 个物理子步；这里挂在 _after_substep 上每 SYNC_EVERY 个子步刷新一次画面（50 Hz），
#     并按 playback 倍速等待，保证看到的速度接近真实（playback=1 为实时，0 为不等待、尽快跑）
#   - 左上角叠加文字：回合、物体、带速、阶段结果（MuJoCo 字体只有 ASCII，所以用英文）
#   - 用户关掉窗口后 is_running() 变成 False，评估脚本据此提前结束
# =============================================================================
SYNC_EVERY = 10


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class SimViewer:
    def __init__(self, env, playback: float = 1.0):
        from mujoco import viewer as mj_viewer          # 不能写 import mujoco.viewer：会把 mujoco 变成局部名
        self.env, self.playback = env, playback
        self.handle = mj_viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
        c, cam = env.cfg, self.handle.cam
        cam.lookat[:] = [c.belt_center_x, 0.05, c.belt_surface_z + 0.10]
        cam.distance, cam.azimuth, cam.elevation = 1.25, -46.0, -24.0
        self._dt = SYNC_EVERY * c.sim_timestep
        self._t_last = time.perf_counter()
        self._orig_after = env._after_substep

        def hooked(k: int):
            self._orig_after(k)
            if k % SYNC_EVERY == 0:
                self.sync()
        env._after_substep = hooked

    def is_running(self) -> bool:
        return self.handle.is_running()

    def sync(self):
        if not self.handle.is_running():
            return
        self.handle.sync()
        if self.playback > 0:
            wait = self._dt / self.playback - (time.perf_counter() - self._t_last)
            if wait > 0:
                time.sleep(wait)
        self._t_last = time.perf_counter()

    def set_text(self, left: str, right: str = ""):
        if self.handle.is_running():
            self.handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))

    def pause(self, seconds: float):
        """Hold the last frame (e.g. between episodes) while keeping the window responsive."""
        t_end = time.perf_counter() + seconds / max(self.playback, 1e-9) if self.playback > 0 else 0
        while time.perf_counter() < t_end and self.handle.is_running():
            self.handle.sync()
            time.sleep(0.03)

    def close(self):
        self.env._after_substep = self._orig_after
        self.handle.close()
        # viewer 在自己的线程里释放 OpenGL；主线程立刻退出解释器会在释放途中崩溃（core dump），等它关完
        t_end = time.perf_counter() + 2.0
        while self.handle.is_running() and time.perf_counter() < t_end:
            time.sleep(0.02)
        time.sleep(0.3)


def belt_speed(env, steps: int) -> float:
    """Mean belt (target) speed over the first ``steps`` control steps of the current episode."""
    n = min(max(steps, 1) * env.n_substeps, len(env.traj.vel))
    return float(np.linalg.norm(env.traj.vel[:n], axis=1).mean())
