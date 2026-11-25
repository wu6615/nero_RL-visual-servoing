"""SB3 VecEnv on top of the batched Isaac env (works with any object offering the small interface below)."""
from __future__ import annotations

import time

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .core.observation import action_space, observation_space

# =============================================================================
# 【说明】自己写一层而不用 Isaac Lab 的 Sb3VecEnvWrapper：SAC 在"超时结束"时要用真正的终止观测做自举，
#   而 DirectRLEnv 返回的是重置之后的观测。NeroGraspEnv 在重置前留了一份 terminal_obs，这里交给 SB3。
#   存下来的策略观测/动作空间和 MuJoCo 版一样，所以能直接给部署代码用（见 scripts/step6_export_policy.py）。
# =============================================================================


class IsaacSb3VecEnv(VecEnv):
    def __init__(self, env, num_grasps: int, num_points: int):
        self.env = env
        super().__init__(env.num_envs, observation_space(num_grasps, num_points), action_space())
        self.ret, self.len, self.t0 = np.zeros(self.num_envs), np.zeros(self.num_envs, int), time.time()
        self._actions = None

    @staticmethod
    def _np(obs) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in obs.items()}

    def reset(self):
        obs, _ = self.env.reset()
        self.ret[:], self.len[:] = 0, 0
        return self._np(obs["policy"])

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        import torch
        obs, rew, term, trunc, _ = self.env.step(torch.as_tensor(self._actions, device=self.env.device, dtype=torch.float32))
        rew, term, trunc = rew.detach().cpu().numpy().astype(np.float64), term.cpu().numpy(), trunc.cpu().numpy()
        done = term | trunc
        self.ret += rew
        self.len += 1
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        ids = np.flatnonzero(done)
        if len(ids):
            last = self._np({k: v[ids] for k, v in self.env.terminal_obs.items()})
            ok = self.env.task_info["success"][ids].cpu().numpy()
            for j, i in enumerate(ids):
                infos[i] = {"terminal_observation": {k: v[j] for k, v in last.items()}, "TimeLimit.truncated": bool(trunc[i] and not term[i]),
                            "is_success": bool(ok[j]),
                            "episode": {"r": float(self.ret[i]), "l": int(self.len[i]), "t": round(time.time() - self.t0, 2)}}
            self.ret[ids], self.len[ids] = 0, 0
        return self._np(obs["policy"]), rew.astype(np.float32), done, infos

    def close(self):
        self.env.close()

    def get_attr(self, name, indices=None):
        return [getattr(self.env, name, None)] * len(self._indices(indices))

    def set_attr(self, name, value, indices=None):
        setattr(self.env, name, value)

    def env_method(self, name, *args, indices=None, **kw):
        raise NotImplementedError(name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._indices(indices))

    def _indices(self, indices):
        return range(self.num_envs) if indices is None else np.atleast_1d(indices)
