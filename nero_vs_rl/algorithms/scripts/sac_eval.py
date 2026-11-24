"""Evaluate a trained SAC policy or the scripted PBVS baseline (GAP-RL ``sac_*_dynamic_eval.py``)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ... import TRAIN_CONFIG
from ...envs.dynamic_grasp import DynamicGraspEnv
from ...envs.viewer import SimViewer, belt_speed, display_available
from ...utils.config import SITE_BELT_SPEED_MAX, load_config
from ..scripted_pbvs import ScriptedGraspPolicy

# 结果按带速分档统计（cm/s）
SPEED_BANDS = [(0, 8), (8, 14), (14, 20), (20, 30), (30, 40), (40, 100)]


def run_episodes(env: DynamicGraspEnv, policy, episodes: int, seed: int, frames: list | None = None,
                 viewer: SimViewer | None = None, label: str = ""):
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        v_belt = belt_speed(env, env.cfg.max_steps)
        ok = sum(r["outcome"] == "success" for r in results)
        if viewer is not None:
            if not viewer.is_running():
                print("图形界面已关闭，提前结束评估")
                break
            viewer.set_text(f"{label}\nepisode {ep + 1}/{episodes}  {info['obj_type']}  belt {v_belt * 100:.1f} cm/s",
                            f"\n\nsuccess so far {ok}/{ep}")
        ret, steps, done, first_success = 0.0, 0, False, None
        while not done:
            action = policy(obs, info)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            steps += 1
            done = term or trunc
            if info["is_success"] and first_success is None:
                first_success = steps
            if frames is not None and ep == 0:
                frames.append(env.render())
        # 与 GAP-RL 评估一致：回合中任一时刻成功即计成功，并记录首次成功步数（完成时间）
        outcome = "success" if first_success is not None else "unstable" if info.get("sim_unstable") else \
            "fell" if info.get("fell") else "missed" if info.get("missed") else "timeout"
        results.append(dict(episode=ep, obj=info["obj_type"], ret=ret, steps=steps, outcome=outcome,
                            first_success_step=first_success, held_at_end=bool(info["is_success"]),
                            belt_speed=v_belt))
        t_succ = f"{first_success * env.control_dt:4.1f}s" if first_success else "   - "
        print(f"[{ep:03d}] {info['obj_type']:8s} 带速 {v_belt * 100:5.1f} cm/s  {outcome:8s} 首次成功 {t_succ} "
              f"return={ret:7.2f}")
        if viewer is not None:
            viewer.set_text(f"{label}\nepisode {ep + 1}/{episodes}  {info['obj_type']}  belt {v_belt * 100:.1f} cm/s",
                            f"\n\n>>> {outcome.upper()}  ({ok + (outcome == 'success')}/{ep + 1})")
            viewer.pause(1.0)
    return results


def speed_table(results: list[dict]) -> list[str]:
    lines = []
    for lo, hi in SPEED_BANDS:
        rs = [r for r in results if lo <= r["belt_speed"] * 100 < hi]
        if rs:
            n_ok = sum(r["outcome"] == "success" for r in rs)
            c = Counter(r["outcome"] for r in rs)
            lines.append(f"  带速 {lo:>2}～{hi if hi < 100 else '':<3} cm/s：{n_ok}/{len(rs)} 成功（{n_ok / len(rs):.0%}）  {dict(c)}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="评估 SAC 策略或脚本 PBVS 基线（仿真）")
    ap.add_argument("--config", default=str(TRAIN_CONFIG), help="默认现场实测配置 dc1_site.yaml；占位对照用 default.yaml")
    ap.add_argument("--model", default=None, help="SAC 模型 .zip；不给则用脚本基线")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", default=None, help="保存第一回合视频（.mp4/.gif，需要 imageio）")
    ap.add_argument("--output", default=None, help="结果 JSON")
    ap.add_argument("--no-viewer", action="store_true", help="不打开 MuJoCo 图形界面（默认有显示器就打开）")
    ap.add_argument("--playback", type=float, default=1.0, help="图形界面播放倍速：1 实时，2 两倍速，0 不等待")
    args = ap.parse_args(argv)

    env_cfg, _ = load_config(args.config)
    print(f"配置: {args.config}  带速范围 {env_cfg.trajectory.speed[0] * 100:.0f}～{env_cfg.trajectory.speed[1] * 100:.0f} cm/s")
    env = DynamicGraspEnv(env_cfg, render_mode="rgb_array" if args.video else None)
    if args.model:
        from stable_baselines3 import SAC
        model = SAC.load(args.model, device="auto")
        policy = lambda obs, info: model.predict(obs, deterministic=True)[0]  # noqa: E731
        label = f"SAC {Path(args.model).parent.parent.name if Path(args.model).parent.name == 'best' else Path(args.model).stem}"
    else:
        policy = ScriptedGraspPolicy(env_cfg)
        label = "scripted PBVS baseline"
    viewer = None
    if not args.no_viewer:
        if display_available():
            viewer = SimViewer(env, playback=args.playback)
            print("已打开 MuJoCo 图形界面（关掉窗口会提前结束评估；鼠标左键转视角、右键平移、滚轮缩放）")
        else:
            print("没有显示器（DISPLAY 为空），不打开图形界面")
    frames = [] if args.video else None
    try:
        results = run_episodes(env, policy, args.episodes, args.seed, frames, viewer, label)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
    if not results:
        return None
    counts = Counter(r["outcome"] for r in results)
    rate = counts["success"] / len(results)
    succ_t = [r["first_success_step"] / env_cfg.control_freq for r in results if r["first_success_step"]]
    held = np.mean([r["held_at_end"] for r in results])
    print(f"成功率 {rate:.1%}  回合末仍抓持 {held:.1%}  平均完成时间 {np.mean(succ_t) if succ_t else float('nan'):.2f}s  "
          f"结果分布 {dict(counts)}  平均回报 {np.mean([r['ret'] for r in results]):.2f}")
    for line in speed_table(results):
        print(line)
    site = [r for r in results if r["belt_speed"] <= SITE_BELT_SPEED_MAX]
    if site:
        n_ok = sum(r["outcome"] == "success" for r in site)
        print(f"现场带速（≤ {SITE_BELT_SPEED_MAX * 100:.0f} cm/s）：{n_ok}/{len(site)} 成功（{n_ok / len(site):.1%}）")
    if args.output:
        Path(args.output).write_text(json.dumps(dict(success_rate=rate, episodes=results), indent=2,
                                                ensure_ascii=False), encoding="utf-8")
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=env_cfg.control_freq)
        print(f"视频: {args.video}")
    return dict(success_rate=rate, episodes=results)


if __name__ == "__main__":
    main()
