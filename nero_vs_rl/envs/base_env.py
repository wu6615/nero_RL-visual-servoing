"""MuJoCo base env for the NERO7: model loading, interpolated control stepping, rendering.

Counterpart of GAP-RL ``envs/base_env.py``; tasks subclass it and fill in the
scene, episode initialisation, observation, evaluation and reward hooks.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .. import SCENE_XML
from ..agents.controllers import EEDeltaPoseController
from ..agents.robots import Nero7Agent
from ..utils.config import EnvConfig

# =============================================================================
# 【说明】一个控制步（10 Hz）内部发生的事：
#   step(action)
#     ├─ controller.compute：动作 → TCP 目标位姿 → IK → 关节目标 q_tgt、夹爪目标
#     ├─ _simulate：n_substeps 个物理子步（默认 0.002 s × 50）
#     │     每子步把关节指令从上一目标线性插值到 q_tgt（对应真机 100 Hz 插值发送），
#     │     并调用 _before_substep / _after_substep 两个钩子（任务在这里驱动物体、查接触）
#     ├─ get_obs → evaluate → compute_dense_reward → check_termination
#     └─ 返回 gymnasium 五元组
# 生命周期与 GAP-RL BaseEnv 一致：__init__ → _load_scene；reset → _initialize_episode。
# =============================================================================


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, config: EnvConfig | dict | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config if isinstance(config, EnvConfig) else EnvConfig.from_dict(config)
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.model.opt.timestep = self.cfg.sim_timestep
        self.n_substeps = int(round(1.0 / (self.cfg.control_freq * self.cfg.sim_timestep)))
        self.control_dt = self.n_substeps * self.cfg.sim_timestep
        self.agent = Nero7Agent(self.model, self.cfg.gripper_max_q)
        self._load_scene()
        self.data = mujoco.MjData(self.model)
        self.controller = EEDeltaPoseController(self.agent, self.cfg, self.control_dt)
        self.q_home = self._solve_home()
        self.controller.ik.q_posture = self.q_home.copy()
        self.step_count = 0
        self.t = 0.0
        self.prev_action = np.zeros(7, dtype=np.float32)
        self._renderer = self._viewer = None

    # ---------------------------------------------------------------- hooks
    def _load_scene(self):
        """Edit self.model (positions, sizes, camera) before MjData is created."""

    def _home_tcp_target(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _initialize_episode(self, options: dict | None):
        raise NotImplementedError

    def _before_substep(self, k: int):
        """k = global physics step index (from episode start) about to be simulated."""

    def _after_substep(self, k: int):
        """Called after each mj_step."""

    def get_obs(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def evaluate(self) -> dict[str, Any]:
        raise NotImplementedError

    def compute_dense_reward(self, info: dict[str, Any]) -> float:
        raise NotImplementedError

    def check_termination(self, info: dict[str, Any]) -> bool:
        return bool(info.get("success", False))

    # ------------------------------------------------------------- lifecycle
    def _solve_home(self) -> np.ndarray:
        """IK for the home TCP pose from many fixed seeds; prefer postures far from joint limits."""
        p, R = self._home_tcp_target()
        ik = self.controller.ik
        rng = np.random.default_rng(7)                       # 固定种子：所有进程得到同一个 home
        span = ik.hi - ik.lo
        best, best_cost = None, np.inf
        seeds = [np.clip(np.asarray(self.cfg.home_seed_joint_rad, float), ik.lo, ik.hi)]
        seeds += [seeds[0] + rng.normal(0, 0.3, 7) for _ in range(23)]
        seeds += [ik.lo + span * rng.uniform(0.2, 0.8, 7) for _ in range(24)]
        for seed in seeds:
            seed = np.clip(seed, ik.lo, ik.hi)
            ik.q_posture = seed
            q, pe, re = ik.solve(seed, p, R, iters=200)
            if pe > 2e-3 or re > 2e-2:
                continue
            margin = np.minimum(q - ik.lo, ik.hi - q) / span    # 离限位的相对距离，0.5 为正中
            cost = float(np.sum((0.5 - margin) ** 2))
            if cost < best_cost:
                best, best_cost = q, cost
        if best is None:
            raise RuntimeError("初始位姿 IK 未收敛，检查 belt_* / init_tcp_* 配置")
        return best

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count, self.t = 0, 0.0
        self.prev_action = np.zeros(7, dtype=np.float32)
        self._initialize_episode(options)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset(self.data)
        obs = self.get_obs()
        return obs, self.evaluate()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_tgt, grip = self.controller.compute(self.data, a)
        bad0 = self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        self._simulate(q_tgt, grip)
        self.step_count += 1
        self.prev_action = a.astype(np.float32)
        obs = self.get_obs()
        info = self.evaluate()
        # 数值发散时 MuJoCo 会自动重置状态，继续跑等于用垃圾数据训练 → 直接结束回合
        info["sim_unstable"] = bool(self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > bad0)
        reward = 0.0 if info["sim_unstable"] else self.compute_dense_reward(info)
        terminated = info["sim_unstable"] or self.check_termination(info)
        truncated = (self.step_count >= self.cfg.max_steps) and not terminated
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _simulate(self, q_tgt: np.ndarray, grip: float):
        d = self.data
        q_start = d.ctrl[self.agent.act_arm].copy()
        k0 = int(round(self.t / self.cfg.sim_timestep))
        for i in range(self.n_substeps):
            self.agent.set_ctrl(d, q_start + (q_tgt - q_start) * (i + 1) / self.n_substeps, grip)
            self._before_substep(k0 + i + 1)
            mujoco.mj_step(self.model, d)
            self._after_substep(k0 + i + 1)
        self.t += self.control_dt

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer   # 不能写 import mujoco.viewer：会把 mujoco 变成局部名
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render().copy()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
