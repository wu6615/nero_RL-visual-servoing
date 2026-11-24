"""Visualise target trajectories and a scripted PBVS rollout (GAP-RL ``examples/test_env_trajs.py``).

  python -m nero_vs_rl.examples.test_env_trajs --mode line --viewer      # MuJoCo 交互窗口
  python -m nero_vs_rl.examples.test_env_trajs --mode bezier2d --video /home/yunfeiwu/adddisk1/MY_RL_test/simulation/demo.mp4
"""
from __future__ import annotations

import argparse
import time

from ..algorithms.scripted_pbvs import ScriptedGraspPolicy
from ..envs.dynamic_grasp import DynamicGraspEnv
from ..utils.config import EnvConfig


def main(argv=None):
    ap = argparse.ArgumentParser(description="可视化物体轨迹 + 脚本 PBVS 抓取（仿真，不连设备）")
    ap.add_argument("--mode", default="line", choices=["line", "bezier2d", "static"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true", help="打开 MuJoCo 交互窗口（需要显示器）")
    ap.add_argument("--video", default=None, help="保存 overview 相机视频 .mp4/.gif")
    args = ap.parse_args(argv)

    cfg = EnvConfig.from_dict({"trajectory": {"mode": args.mode}})
    env = DynamicGraspEnv(cfg, render_mode="human" if args.viewer else "rgb_array")
    policy = ScriptedGraspPolicy(cfg)
    frames = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        policy.reset()
        done, ever = False, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, info))
            done = term or trunc
            ever |= info["is_success"]
            if args.viewer:
                env.render()
                time.sleep(env.control_dt)
            elif args.video:
                frames.append(env.render())
        print(f"回合 {ep}: {info['obj_type']} 成功过={ever} 回合末仍抓持={info['is_success']} 阶段={policy.phase}")
    env.close()
    if args.video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=cfg.control_freq)
        print(f"视频: {args.video}")


if __name__ == "__main__":
    main()
