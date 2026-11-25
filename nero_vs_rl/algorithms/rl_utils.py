"""SB3 glue: vectorised env construction (GAP-RL ``algorithms/rl_utils.py`` counterpart)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ..envs.dynamic_grasp import DynamicGraspEnv
from ..utils.config import SITE_BELT_SPEED_MAX, EnvConfig

# 【说明】GAP-RL 用自己的 vector/ 封装 SAPIEN 并行环境；MuJoCo 环境很轻，
# 直接用 SB3 的 SubprocVecEnv（每个进程一个环境）即可，所以这里没有 vector/ 目录。
#
# 【说明】每个环境子进程只用 1 个线程：
#   环境里只有小矩阵运算（6x6 求解、几十个点的变换），多线程 BLAS/torch 不会更快，
#   8 个进程各开一堆线程反而争抢 CPU。2026-11-22 本机实测（8 环境，含梯度更新的稳态速度）：
#   不限线程：每个子进程占 120%~230% CPU，88 步/s；限成 1 线程：269 步/s（约 3 倍）。
#   forkserver 子进程是全新的解释器，所以在创建前设环境变量即可生效；主进程（GPU 训练）不受影响。
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _single_thread_worker():
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def make_vec_env(env_cfg: EnvConfig, n_envs: int, seed: int, subproc: bool = True):
    def factory(rank: int):
        def _init():
            if subproc and n_envs > 1:
                _single_thread_worker()
            env = DynamicGraspEnv(env_cfg)
            env.reset(seed=seed + rank)
            return env
        return _init
    fns = [factory(i) for i in range(n_envs)]
    if subproc and n_envs > 1:
        saved = {k: os.environ.get(k) for k in THREAD_ENV}
        os.environ.update({k: "1" for k in THREAD_ENV})
        try:
            vec = SubprocVecEnv(fns, start_method="forkserver")
        finally:                                      # 只影响子进程，主进程环境变量还原
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    else:
        vec = DummyVecEnv(fns)
    return VecMonitor(vec, info_keywords=("is_success",))


# =============================================================================
# 【说明】训练中的评估与选模型（替代 SB3 EvalCallback，2026-11-23 起）：
#   · 每次评估都跑同一批 n_episodes 个回合（reset(seed=seed0 + i)），不同时刻的模型比的是同一组题，
#     去掉"这次抽到的回合难不难"这部分噪声。EvalCallback 每次抽的回合都不一样，而且只有 20 回合：
#     成功率 85% 时 20 回合的标准误差约 ±8%，v5 曲线后半段 75～95% 来回跳大多是这个噪声。
#   · best_model 按成功率选（并列时看平均回报），不按平均回报选：回报里有接近、抬升等稠密项，
#     高回报不一定是抓得更多。
#   · 评估种子从 seed0（默认 900000）开始，和最终测试用的 20000～20299 不重叠，选模型不偷看测试集。
#   · 另外单独记现场带速（≤ SITE_BELT_SPEED_MAX）的成功率。
#   输出与 EvalCallback 同名：eval/evaluations.npz（timesteps / results / successes / ep_lengths，另加 belt_speeds），
#   TensorBoard 里是 eval/success_rate、eval/site_success_rate、eval/mean_reward。
# =============================================================================
class SuccessEvalCallback(BaseCallback):
    def __init__(self, env_cfg: EnvConfig, eval_freq: int, n_episodes: int, run_dir: Path, seed0: int = 900_000):
        super().__init__()
        self.env = DynamicGraspEnv(env_cfg)
        self.eval_freq, self.n_episodes, self.seed0 = eval_freq, n_episodes, seed0
        self.best_path = Path(run_dir) / "best" / "best_model"
        self.log_path = Path(run_dir) / "eval" / "evaluations.npz"
        self.best = (-1.0, -np.inf)
        self.hist = {k: [] for k in ("timesteps", "results", "successes", "ep_lengths", "belt_speeds")}

    def load_history(self):
        """Resume: take earlier evaluations and the best so far, so a worse model never overwrites best/."""
        if self.log_path.exists():
            z = np.load(self.log_path)
            self.hist = {k: list(z[k]) for k in self.hist}
            for succ, rets in zip(self.hist["successes"], self.hist["results"]):
                self.best = max(self.best, (float(np.mean(succ)), float(np.mean(rets))))

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.evaluate()
        return True

    def run_episode(self, seed: int) -> tuple[float, bool, int, float]:
        env = self.env
        obs, info = env.reset(seed=seed)
        n = min(env.cfg.max_steps * env.n_substeps, len(env.traj.vel))
        speed = float(np.linalg.norm(env.traj.vel[:n], axis=1).mean())
        ret, steps, success, done = 0.0, 0, False, False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret, steps = ret + r, steps + 1
            success |= bool(info["is_success"])
            done = term or trunc
        return ret, success, steps, speed

    def evaluate(self):
        rows = [self.run_episode(self.seed0 + i) for i in range(self.n_episodes)]
        rets, succ, lens, speeds = (np.array(x) for x in zip(*rows))
        rate, site = float(succ.mean()), speeds <= SITE_BELT_SPEED_MAX
        site_rate = float(succ[site].mean()) if site.any() else float("nan")
        self.logger.record("eval/success_rate", rate)
        self.logger.record("eval/site_success_rate", site_rate)
        self.logger.record("eval/mean_reward", float(rets.mean()))
        for k, v in zip(self.hist, (self.num_timesteps, rets, succ, lens, speeds)):
            self.hist[k].append(v)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.log_path, **{k: np.array(v) for k, v in self.hist.items()})
        better = (rate, float(rets.mean())) > self.best
        if better:
            self.best = (rate, float(rets.mean()))
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_path)
        print(f"[评估] {self.num_timesteps} 步：成功率 {rate:.1%}（现场带速 {site_rate:.1%}，{int(site.sum())} 回合）"
              f"，平均回报 {rets.mean():.1f}" + ("  ← 新的最好模型" if better else ""), flush=True)

    def _on_training_end(self):
        self.env.close()
