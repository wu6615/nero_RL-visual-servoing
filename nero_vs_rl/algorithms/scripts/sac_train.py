"""Train SAC on the NERO dynamic grasp env (GAP-RL ``algorithms/scripts/sac_train.py`` counterpart)."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from ... import TRAIN_CONFIG
from ...utils.config import load_config
from ..Networks import GraspPointGroupExtractor
from ..rl_utils import SuccessEvalCallback, make_vec_env


def main(argv=None):
    ap = argparse.ArgumentParser(description="SAC 训练：NERO7 眼在手上动态抓取")
    ap.add_argument("--config", default=str(TRAIN_CONFIG), help="默认现场实测配置 dc1_site.yaml；占位对照用 default.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--total-timesteps", type=int, default=None, help="覆盖配置（冒烟测试用）")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--init-model", default=None,
                    help="从已有模型 .zip 的网络权重继续训练（如占位参数下训好的 v1），步数和日志从 0 开始")
    args = ap.parse_args(argv)

    env_cfg, tr = load_config(args.config)
    print(f"配置: {args.config}")
    if args.total_timesteps is not None:
        tr.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        tr.n_envs = args.n_envs
    run = Path(tr.log_dir) / (args.run_name or time.strftime("sac_%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run / "config.yaml")
    # 【说明】warm start：观测/动作空间与相机参数无关（抓取点在 TCP 系），所以占位参数下训好的网络
    #   可以直接接着在实测参数下训。只搬网络权重、优化器和熵系数；经验池不保存在 .zip 里，从空开始。
    #   learning_starts 期间 SB3 用均匀随机动作采样，已有策略时缩短到 1000 步，尽快用策略本身采数据。
    if args.init_model:
        tr.learning_starts = min(tr.learning_starts, 1000)
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump({"env": env_cfg.to_dict(), "train": vars(tr), "init_model": args.init_model}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    venv = make_vec_env(env_cfg, tr.n_envs, tr.seed)
    model = SAC(
        "MultiInputPolicy", venv,
        learning_rate=tr.learning_rate, buffer_size=tr.buffer_size, batch_size=tr.batch_size,
        gamma=tr.gamma, tau=tr.tau, learning_starts=tr.learning_starts, train_freq=tr.train_freq,
        gradient_steps=tr.gradient_steps, ent_coef=tr.ent_coef, seed=tr.seed, device=tr.device,
        tensorboard_log=str(run / "tb"), verbose=1,
        policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=tr.net_arch),
    )
    if args.init_model:
        model.set_parameters(args.init_model, exact_match=True, device=tr.device)
        print(f"从 {args.init_model} 的权重继续训练")
    callbacks = [CheckpointCallback(max(tr.checkpoint_every // tr.n_envs, 1), str(run / "checkpoints"), "sac")]
    if not args.no_eval:          # 固定一批回合、按成功率选最好模型（见 rl_utils.SuccessEvalCallback）
        callbacks.append(SuccessEvalCallback(env_cfg, max(tr.eval_every // tr.n_envs, 1), tr.eval_episodes, run))
    print(f"训练目录: {run}")
    try:
        model.learn(total_timesteps=tr.total_timesteps, callback=callbacks, progress_bar=False)
    finally:
        model.save(run / "final_model")
        venv.close()
    print(f"已保存: {run / 'final_model.zip'}")


if __name__ == "__main__":
    main()
