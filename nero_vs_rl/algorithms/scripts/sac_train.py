"""Train SAC on the NERO dynamic grasp env (GAP-RL ``algorithms/scripts/sac_train.py`` counterpart).

  python -m nero_vs_rl.algorithms.scripts.sac_train --run-name sac_dc1_v7             # 新训练
  python -m nero_vs_rl.algorithms.scripts.sac_train --run-name sac_dc1_v7 --resume    # 接着暂停处继续（Ctrl+C 暂停）
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from ... import TRAIN_CONFIG
from ...utils.config import EnvConfig, TrainConfig, load_config
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
    ap.add_argument("--resume", action="store_true",
                    help="接着 --run-name 目录里 paused_model.zip（Ctrl+C 时保存）继续：步数、日志、最好模型都接上")
    args = ap.parse_args(argv)
    run_existing = Path(load_config(args.config)[1].log_dir) / (args.run_name or "")
    if args.resume:
        if not args.run_name or not (run_existing / PAUSED_MODEL).exists():
            raise SystemExit(f"--resume 需要 --run-name，且 {run_existing / PAUSED_MODEL} 存在")
        # 用当时展开好的完整配置（resolved_config.yaml），不受之后改 default.yaml / dc1_site.yaml 影响
        data = yaml.safe_load((run_existing / "resolved_config.yaml").read_text(encoding="utf-8"))
        env_cfg, tr = EnvConfig.from_dict(data["env"]), TrainConfig.from_dict(data["train"])
    else:
        env_cfg, tr = load_config(args.config)
    print(f"配置: {args.config}")
    if args.total_timesteps is not None:
        tr.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        tr.n_envs = args.n_envs
    run = Path(tr.log_dir) / (args.run_name or time.strftime("sac_%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        shutil.copy(args.config, run / "config.yaml")
    # 【说明】warm start：观测/动作空间与相机参数无关（抓取点在 TCP 系），所以占位参数下训好的网络
    #   可以直接接着在实测参数下训。只搬网络权重、优化器和熵系数；经验池不保存在 .zip 里，从空开始。
    #   learning_starts 期间 SB3 用均匀随机动作采样，已有策略时缩短到 1000 步，尽快用策略本身采数据。
    if args.init_model:
        tr.learning_starts = min(tr.learning_starts, 1000)
    if not args.resume:
        resolved = {"env": env_cfg.to_dict(), "train": vars(tr), "init_model": args.init_model}
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
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
    if args.resume:
        model = resume_model(run, venv, tr)
    callbacks = [CheckpointCallback(max(tr.checkpoint_every // tr.n_envs, 1), str(run / "checkpoints"), "sac")]
    if not args.no_eval:          # 固定一批回合、按成功率选最好模型（见 rl_utils.SuccessEvalCallback）
        callbacks.append(SuccessEvalCallback(env_cfg, max(tr.eval_every // tr.n_envs, 1), tr.eval_episodes, run))
    if args.resume and not args.no_eval:
        callbacks[-1].load_history()
    print(f"训练目录: {run}")
    remaining = tr.total_timesteps - model.num_timesteps
    try:
        model.learn(total_timesteps=remaining, callback=callbacks, progress_bar=False,
                    reset_num_timesteps=not args.resume, tb_log_name="SAC")
    except KeyboardInterrupt:
        # 【说明】Ctrl+C = 暂停：权重、优化器、熵系数和经验池都存下来，--resume 从这里接着训
        model.save(run / PAUSED_MODEL)
        model.save_replay_buffer(run / PAUSED_BUFFER)
        print(f"\n已暂停在 {model.num_timesteps} 步：{run / PAUSED_MODEL}（+ 经验池）。继续：\n"
              f"python -m nero_vs_rl.algorithms.scripts.sac_train --run-name {run.name} --resume")
        return model
    finally:
        venv.close()
    model.save(run / "final_model")
    print(f"已保存: {run / 'final_model.zip'}")
    return model


PAUSED_MODEL, PAUSED_BUFFER = "paused_model.zip", "paused_replay_buffer.pkl"


def resume_model(run: Path, venv, tr) -> SAC:
    """Load the paused model (and its replay buffer if it was saved) into the fresh vec env."""
    model = SAC.load(run / PAUSED_MODEL, env=venv, device=tr.device, tensorboard_log=str(run / "tb"))
    buf = run / PAUSED_BUFFER
    if buf.exists():
        model.load_replay_buffer(buf)
        print(f"接着 {model.num_timesteps} 步继续，经验池 {model.replay_buffer.size()} 条")
    else:
        # 没有经验池（旧版脚本中断时没存）：先用策略本身再攒一点数据再开始更新。
        # 【说明】SB3 在 learning_starts 之前用均匀随机动作采样，所以只留 2000 步，尽快换回策略
        model.learning_starts = model.num_timesteps + 2000
        print(f"接着 {model.num_timesteps} 步继续；没有保存的经验池，先攒 2000 步再更新")
    return model


if __name__ == "__main__":
    main()
