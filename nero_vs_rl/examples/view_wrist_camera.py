"""Render what the simulated wrist DC1 sees (RGB + depth with blind zone) along a scripted rollout.

  python -m nero_vs_rl.examples.view_wrist_camera        # 保存 PNG 序列到数据盘 wrist/，默认用实测相机参数
无显示器的机器上先 export MUJOCO_GL=egl。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import SIM_OUTPUT_DIR, TRAIN_CONFIG
from ..algorithms.scripted_pbvs import ScriptedGraspPolicy
from ..envs.dynamic_grasp import DynamicGraspEnv
from ..sensors.camera import depth_to_points
from ..utils.config import load_config


def depth_to_rgb(depth: np.ndarray, max_depth: float) -> np.ndarray:
    v = np.clip(depth / max_depth, 0, 1)
    img = (255 * (1 - v)).astype(np.uint8)
    img[depth <= 0] = 0                                     # 无效/盲区显示为黑
    return np.repeat(img[..., None], 3, axis=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="保存仿真腕部相机 RGB / 深度图")
    ap.add_argument("--out", default=str(SIM_OUTPUT_DIR / "wrist"))
    ap.add_argument("--config", default=str(TRAIN_CONFIG), help="默认现场实测配置 dc1_site.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=5, help="每隔几步存一帧")
    args = ap.parse_args(argv)
    import imageio.v2 as imageio

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, _ = load_config(args.config)
    env = DynamicGraspEnv(cfg)
    policy = ScriptedGraspPolicy(cfg)
    obs, info = env.reset(seed=args.seed)
    done = False
    while not done:
        if env.step_count % args.every == 0:
            rgb, depth = env.render_wrist_rgbd()
            pts = depth_to_points(depth, cfg.camera)
            imageio.imwrite(out / f"{env.step_count:03d}_rgb.png", rgb)
            imageio.imwrite(out / f"{env.step_count:03d}_depth.png", depth_to_rgb(depth, 0.6))
            print(f"step {env.step_count:3d} 可见={info['visible']} 有效深度点={len(pts)} 阶段={policy.phase}")
        obs, r, term, trunc, info = env.step(policy(obs, info))
        done = term or trunc
    env.close()
    print(f"已保存到 {out}")


if __name__ == "__main__":
    main()
