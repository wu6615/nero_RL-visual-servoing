"""Joint commands -> the RL policy's 7-d action: the inverse of ``EEDeltaPoseController.target_pose``.

A demonstrated step "the arm was commanded from q_k to q_{k+1} within one control period" becomes the action that
would make the RL controller command the same TCP motion, so demonstrations live in the policy's own action space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...utils.config import EnvConfig
from ...utils.geometry import logmap
from ..deploy.runtime.devices import width_to_q

# =============================================================================
# 【说明】RL 动作的定义（agents/controllers/pd_ee_delta_pose.py）：
#   a[0:3] 平移增量 / max_delta_pos_m，a[3:6] 旋转向量增量 / max_delta_rot_rad，都在 action_frame 下表达：
#          ee   ：p_next = p + R·dpos，      R_next = R·exp(drot)    ⇒ dpos = Rᵀ(p_next − p)，drot = log(Rᵀ·R_next)
#          base ：p_next = p + dpos，        R_next = exp(drot)·R    ⇒ dpos = p_next − p，    drot = log(R_next·Rᵀ)
#   a[6]   夹爪目标开口：−1 闭合，+1 全开；仿真单指行程 q = (a6 + 1)/2 × gripper_max_q，真机宽度 = 2q
#          ⇒ a6 = 2 × q / gripper_max_q − 1，q = width_to_q(NERO 的夹爪指令宽度)
#   p、R 用的是"指令"位姿，不是实测位姿：RL 控制器的目标就是在上一步指令上累积的（同 NERO 的 100 Hz 指令曲线）。
# 超出 [−1, 1] 的分量（NERO 一步走得比 RL 允许的还远）截到边界并计数：示范里这一步 RL 做不到，
#   训练时可以据此决定丢掉、或者放宽 max_delta_*。
# =============================================================================


@dataclass
class DemoAction:
    action: np.ndarray       # (7,) 已截到 [−1, 1]
    raw: np.ndarray          # (7,) 截断前
    clipped: bool


def joints_to_action(fk, q_now: np.ndarray, q_next: np.ndarray, width_next: float, cfg: EnvConfig) -> DemoAction:
    """``fk(q) -> (p, R)`` of the TCP (same model as training); widths in metres as the real gripper reports them."""
    p0, R0 = fk(np.asarray(q_now, float))
    p1, R1 = fk(np.asarray(q_next, float))
    if cfg.action_frame == "ee":
        dpos, drot = R0.T @ (p1 - p0), logmap(R0.T @ R1)
    elif cfg.action_frame == "base":
        dpos, drot = p1 - p0, logmap(R1 @ R0.T)
    else:
        raise ValueError(f"未知 action_frame: {cfg.action_frame}")
    grip = 2.0 * width_to_q(width_next) / cfg.gripper_max_q - 1.0
    raw = np.concatenate([dpos / cfg.max_delta_pos_m, drot / cfg.max_delta_rot_rad, [grip]])
    action = np.clip(raw, -1.0, 1.0)
    return DemoAction(action.astype(np.float32), raw, bool(np.any(np.abs(raw[:6]) > 1.0 + 1e-6)))
