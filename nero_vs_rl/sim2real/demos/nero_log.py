"""Read-only parser for a nero_conveyor_ros2 continuous-picking run directory: per-cycle joint commands, stages, gripper commands.

Only reads files; nothing from the NERO project is imported or modified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# =============================================================================
# 【说明】NERO 工程一次连续抓取的运行目录（runs/continuous_<时间>_<pid>/）里，每件工件一个 cycle_NNN/：
#   execution_trace.npz          100 Hz 实际发出的关节指令 dispatched_q_rad，发送时刻 dispatch_start_monotonic_s，
#                                每个点属于哪一段 sample_stage（接近拦截 / 原地保持 / 搬运 / 返回 ...）
#   execution_trace_summary.json gripper_commands：每次夹爪指令的时刻和宽度（0 = 闭合，0.1 = 张开）
#   events.json                  本件的事件；grip_confirmed_from_width_force = NERO 按开口和力判定"夹住了"
#   report.json                  本件的状态、周期时长
# 带速不在 cycle_*/ 里：NERO 侧录制结束时从它的运行日志抽成 belt_speed.json（见 NERO wrist_dc1/run_logs.py），
#   per_target 按目标身份 (track_generation, track_id) 对应到每件（execution_trace_summary.json 的 target_identity）。
# 时间都是 time.monotonic（同一台电脑上所有进程共用一个时钟），所以能和 NERO 的腕部录制（<运行目录>/wrist_dc1/）直接对齐。
#
# 示范数据只要"接近 → 闭合"这一段：名字里带 intercept 的那一段（现场主线是 smooth_intercept_follow_brake：
#   接入 → 计划接触时刻闭爪 → 跟随减速）。之后的原地保持、搬运入框、返回等待位不是 RL 的任务。
# =============================================================================
INTERCEPT_KEY = "intercept"
GRIP_CONFIRMED_EVENT = "grip_confirmed_from_width_force"
CLOSED_WIDTH_M = 0.05            # 夹爪指令宽度小于它算"闭合"


@dataclass
class Stage:
    name: str
    t_start: float
    t_end: float


@dataclass
class Cycle:
    index: int
    path: Path
    t: np.ndarray                     # (N,) 指令发送时刻（monotonic）
    q: np.ndarray                     # (N, 7) 发出的关节指令
    stages: list[Stage]
    gripper: list[tuple[float, float]]  # [(时刻, 宽度 m)]
    grip_confirmed: bool
    report: dict = field(default_factory=dict)
    target: tuple | None = None       # (track_generation, track_id)，用来对上 belt_speed.json 的 per_target

    @property
    def intercept(self) -> Stage | None:
        return next((s for s in self.stages if INTERCEPT_KEY in s.name), None)

    @property
    def t_close(self) -> float | None:
        """First close command during the intercept stage (None if the gripper never closed there)."""
        st = self.intercept
        if st is None:
            return None
        return next((t for t, w in self.gripper if w < CLOSED_WIDTH_M and st.t_start <= t <= st.t_end), None)


def _stages(t: np.ndarray, names: np.ndarray) -> list[Stage]:
    out, start = [], 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[start]:
            out.append(Stage(str(names[start]), float(t[start]), float(t[i - 1])))
            start = i
    return out


def load_cycle(path: Path, index: int) -> Cycle | None:
    path = Path(path)
    trace = path / "execution_trace.npz"
    if not trace.exists():
        return None
    z = np.load(trace, allow_pickle=True)
    t, q = np.asarray(z["dispatch_start_monotonic_s"], float), np.asarray(z["dispatched_q_rad"], float)
    order = np.argsort(t)
    t, q, names = t[order], q[order], np.asarray(z["sample_stage"])[order]
    summary = json.loads((path / "execution_trace_summary.json").read_text(encoding="utf-8"))
    gripper = sorted((float(g["sdk_start_monotonic"]), float(g["width_m"]))
                     for g in summary.get("gripper_commands", []) if g.get("succeeded", True))
    events_file, report_file = path / "events.json", path / "report.json"
    events = json.loads(events_file.read_text(encoding="utf-8")) if events_file.exists() else []
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    confirmed = any(e.get("event") == GRIP_CONFIRMED_EVENT for e in events)
    ident = summary.get("target_identity") or {}
    target = (ident.get("track_generation"), ident.get("track_id")) if ident else None
    return Cycle(index, path, t, q, _stages(t, names), gripper, confirmed, report, target)


def load_run(run_dir: str | Path) -> list[Cycle]:
    cycles = []
    for i, d in enumerate(sorted(Path(run_dir).glob("cycle_*"))):
        c = load_cycle(d, i)
        if c is not None:
            cycles.append(c)
    if not cycles:
        raise SystemExit(f"{run_dir} 里没有 cycle_*/execution_trace.npz：不是 NERO 连续抓取的运行目录？")
    return cycles


# =============================================================================
# 【说明】把所有周期的指令拼成一条时间线，给示范转换按任意时刻取"那一刻机械臂被要求在哪"：
#   q_at(t)      t 落在某段 100 Hz 指令里 → 线性插值；不在任何一段里（等待位静止、两件之间）→ None，
#                调用方改用实测关节（机械臂不动时指令 = 实测）
#   width_at(t)  t 之前最后一次夹爪指令的宽度；还没有任何指令时按张开（0.1 m）
# =============================================================================
class CommandTimeline:
    GAP_S = 0.05                     # 相邻两个指令点隔得比它远 → 中间不插值（两段之间）

    def __init__(self, cycles: list[Cycle]):
        t = np.concatenate([c.t for c in cycles])
        q = np.concatenate([c.q for c in cycles])
        order = np.argsort(t)
        self.t, self.q = t[order], q[order]
        g = sorted(x for c in cycles for x in c.gripper)
        self.g_t = np.array([x[0] for x in g])
        self.g_w = np.array([x[1] for x in g])

    def q_at(self, t: float) -> np.ndarray | None:
        i = int(np.searchsorted(self.t, t))
        if i == 0 or i >= len(self.t):
            return None
        t0, t1 = self.t[i - 1], self.t[i]
        if t1 - t0 > self.GAP_S:
            return None
        a = (t - t0) / max(t1 - t0, 1e-9)
        return (1 - a) * self.q[i - 1] + a * self.q[i]

    def width_at(self, t: float, open_width: float = 0.1) -> float:
        i = int(np.searchsorted(self.g_t, t, side="right"))
        return open_width if i == 0 else float(self.g_w[i - 1])


def load_belt_speed(path: str | Path) -> dict[tuple, dict]:
    """belt_speed.json -> {(track_generation, track_id): {speed_m_s, velocity_base_m_s, source, ...}}; {} if absent."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(r.get("track_generation"), r.get("track_id")): r for r in data.get("per_target", [])}
