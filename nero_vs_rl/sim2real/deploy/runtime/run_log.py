"""Closed-loop run log: every 10 Hz tick in ticks.jsonl, plus depth frames in the step1 record format (replayable by step2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import RecordWriter, _np_default, write_json

# =============================================================================
# 【说明】一次运行一个目录（真机在 REAL_DATA_DIR，仿真干跑在 SIM_OUTPUT_DIR/closed_loop_sim）：
#   ticks.jsonl      每拍一行：阶段、实测/指令/目标关节、夹爪、策略动作、感知结果、安全问题、计算耗时
#   episodes.json    每回合一条汇总（结果、原因、用时）
#   meta.json + frames.jsonl + depth/ + color/   与 step1 录制同格式（--no-frames 时不存）
#                    ⇒ 闭环跑过的数据可以直接用 step2_replay 离线回放、和策略"当时看到的"逐帧对照
#   joints_hist.npz  100 Hz 关节历史（step3 测延迟、事后分析跟随误差用）
# =============================================================================
class RunLog:
    def __init__(self, out: Path, meta: dict, save_frames: bool = True):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if save_frames:
            (self.out / "depth").mkdir(exist_ok=True)
            (self.out / "color").mkdir(exist_ok=True)
            self.writer = RecordWriter(self.out, meta)
        else:
            write_json(self.out / "meta.json", meta)
        self._ticks = open(self.out / "ticks.jsonl", "a", encoding="utf-8")
        self.episodes: list[dict] = []
        self._last_stamp = None
        self._hist: list[tuple[float, np.ndarray]] = []

    def tick(self, row: dict, frame, arm):
        self._ticks.write(json.dumps(row, ensure_ascii=False, default=_np_default) + "\n")
        if self.writer is not None and frame is not None and frame.stamp != self._last_stamp:
            self._last_stamp = frame.stamp
            self.writer.write(frame.stamp, arm.joints_at(frame.stamp), frame.depth, frame.color,
                              episode=row["ep"], phase=row["phase"])
        if hasattr(arm, "history"):
            since = self._hist[-1][0] if self._hist else -np.inf
            self._hist += arm.history(since)

    def episode(self, summary: dict):
        self.episodes.append(summary)
        write_json(self.out / "episodes.json", self.episodes)

    def close(self):
        self._ticks.close()
        if self.writer is not None:
            self.writer.close()
        if self._hist:
            np.savez_compressed(self.out / "joints_hist.npz", t=np.array([h[0] for h in self._hist]),
                                q=np.array([h[1] for h in self._hist]))
