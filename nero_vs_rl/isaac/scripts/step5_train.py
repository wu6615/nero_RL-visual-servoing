"""Step 5 (server) - SAC training on the batched env (SB3, same network as the MuJoCo side).

  python -m nero_vs_rl.isaac.scripts.step5_train --headless --run-name isaac_v1
  python -m nero_vs_rl.isaac.scripts.step5_train --headless --run-name isaac_v1 --resume     # Ctrl+C 暂停之后接着训
"""
from __future__ import annotations

import argparse
import sys
from collections import deque

from ._app import finish, start

PAUSED, BUFFER = "paused_model.zip", "paused_replay_buffer.pkl"


def best_callback(run, window: int):
    """Keep the model with the best success rate over the last `window` finished episodes."""
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback

    class Best(BaseCallback):
        def __init__(self):
            super().__init__()
            self.hist, self.best, self.file = deque(maxlen=window), -1.0, run / "best" / "success_rate.txt"
            if self.file.exists():
                self.best = float(self.file.read_text())

        def _on_step(self):
            self.hist.extend(i["is_success"] for i in self.locals["infos"] if "is_success" in i)
            if len(self.hist) == window and self.n_calls % 200 == 0:
                rate = float(np.mean(self.hist))
                self.logger.record("rollout/success_window", rate)
                if rate > self.best:
                    self.best = rate
                    self.file.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(run / "best" / "best_model")
                    self.file.write_text(f"{rate:.4f}")
                    print(f"[训练] {self.num_timesteps} 步：最近 {window} 回合成功率 {rate:.1%}  ← 新的最好模型", flush=True)
            return True

    return Best()


def main():
    ap = argparse.ArgumentParser(description="Isaac 版 SAC 训练")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--total-timesteps", type=int, default=None, help="覆盖 isaac.yaml（冒烟测试用）")
    args, app, tc, ic = start(ap)
    code = 1
    try:
        import yaml
        from dataclasses import asdict
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback
        from ...algorithms.Networks import GraspPointGroupExtractor
        from .. import OUTPUT_DIR
        from ..sb3_vec_env import IsaacSb3VecEnv
        from ..sim.grasp_env import make_env
        t = ic.train
        run = OUTPUT_DIR / args.run_name
        if args.resume != (run / PAUSED).exists():
            raise SystemExit(f"--resume 和 {run / PAUSED} 是否存在对不上：暂停过才加 --resume，新训练换一个 --run-name")
        run.mkdir(parents=True, exist_ok=True)
        (run / "resolved_config.yaml").write_text(yaml.safe_dump({"task": tc.to_dict(), "isaac": asdict(ic)}, allow_unicode=True, sort_keys=False),
                                                  encoding="utf-8")
        env = make_env(tc, ic, args.num_envs, args.device, args.seed)
        venv = IsaacSb3VecEnv(env, tc.grasp_obs.num_grasps, tc.grasp_obs.num_points)
        if args.resume:
            model = SAC.load(run / PAUSED, env=venv, device=args.device, tensorboard_log=str(run / "tb"))
            model.load_replay_buffer(run / BUFFER)
        else:
            model = SAC("MultiInputPolicy", venv, learning_rate=t.learning_rate, buffer_size=t.buffer_size, batch_size=t.batch_size,
                        gamma=t.gamma, tau=t.tau, learning_starts=t.learning_starts, train_freq=1, gradient_steps=t.gradient_steps,
                        seed=args.seed, device=args.device, tensorboard_log=str(run / "tb"), verbose=1,
                        policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=list(t.net_arch)))
        total = args.total_timesteps or t.total_timesteps
        cbs = [CheckpointCallback(max(t.checkpoint_every // venv.num_envs, 1), str(run / "checkpoints"), "sac"), best_callback(run, t.success_window)]
        print(f"训练目录: {run}  并行环境 {venv.num_envs}  从 {model.num_timesteps} 步到 {total} 步")
        try:
            model.learn(total - model.num_timesteps, callback=cbs, reset_num_timesteps=not args.resume, tb_log_name="SAC")
            model.save(run / "final_model")
            (run / PAUSED).unlink(missing_ok=True)
            print(f"已保存 {run / 'final_model.zip'}")
        except KeyboardInterrupt:
            model.save(run / PAUSED)
            model.save_replay_buffer(run / BUFFER)
            print(f"\n已暂停在 {model.num_timesteps} 步。继续：加 --resume 重跑同一条命令")
        code = 0
    finally:
        finish(app, code)


if __name__ == "__main__":
    sys.exit(main())
