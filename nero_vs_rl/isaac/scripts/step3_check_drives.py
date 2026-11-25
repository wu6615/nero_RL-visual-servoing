"""Step 3 (server) - joint drive response against the real arm; tune isaac.yaml `drive` until it matches.

  python -m nero_vs_rl.isaac.scripts.step3_check_drives --headless --num-envs 16
"""
from __future__ import annotations

import argparse
import sys

from ._app import finish, start

# =============================================================================
# 【说明】每个关节按 10 Hz 给正弦目标（幅度 0.15 rad，周期 1.6 s），和环境里一样在控制步内线性插值，
#   测"实测相对插值后的指令"的最佳时间平移。真机（2026-11-24 全速闭环）J1～J7 是 90～100 ms，MuJoCo 版 80～110 ms
#   （理论值 = damping / stiffness = 89～113 ms）。
#   验收：80～120 ms。偏大就加 stiffness 或减 damping。夹爪：全开 → 全闭指令后 0.2 s 内开口 < 1.5 cm。
# =============================================================================


def main():
    args, app, tc, ic = start(argparse.ArgumentParser(description="关节和夹爪的跟随滞后"))
    code = 1
    try:
        import numpy as np
        import torch
        from ..sim.grasp_env import make_env
        env = make_env(tc, ic, args.num_envs or 16, args.device, args.seed)
        env.reset()
        env.task.attached[:] = False
        B, dt, dec = env.num_envs, env.physics_dt, env.cfg.decimation
        q0 = env.task.q_home.expand(B, 7).clone()
        ok = True
        for j in range(7):
            env.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=env.arm_ids)
            cmd, meas, prev = [], [], q0.clone()
            for k in range(48):
                tgt = q0.clone()
                tgt[:, j] += 0.15 * np.sin(2 * np.pi * (k + 1) / 16)
                for s in range(dec):
                    c = prev + (tgt - prev) * (s + 1) / dec
                    env.robot.set_joint_position_target(c, joint_ids=env.arm_ids)
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt)
                    cmd.append(float(c[0, j]))
                    meas.append(float(env.robot.data.joint_pos[0, env.arm_ids[j]]))
                prev = tgt
            cmd, meas = np.array(cmd), np.array(meas)
            t = np.arange(len(cmd)) * dt
            keep = t > 1.6
            lags = np.arange(0, 0.4, 0.005)
            rms = [np.sqrt(np.mean((meas[keep] - np.interp(t[keep] - lag, t, cmd)) ** 2)) for lag in lags]
            lag = lags[int(np.argmin(rms))]
            good = 0.08 <= lag <= 0.12
            ok &= good
            print(f"  {'✓' if good else '✗'} J{j + 1} 滞后 {lag * 1000:.0f} ms，残差 {min(rms):.4f} rad")
        g = env.task_cfg.gripper_max_q
        env.robot.set_joint_position_target(torch.zeros(B, 2, device=env.device), joint_ids=env.grip_ids)
        for _ in range(int(0.2 / dt)):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt)
        w = 2 * float(env._state().grip.mean()) * 100
        ok &= w < 1.5
        print(f"  {'✓' if w < 1.5 else '✗'} 夹爪 {2 * g * 100:.0f} cm → 全闭，0.2 s 后开口 {w:.1f} cm（真机 0.8 cm）")
        print("全部通过" if ok else "有没通过的项：改 isaac.yaml 的 drive 再跑")
        code = 0 if ok else 1
        env.close()
    finally:
        finish(app, code)


if __name__ == "__main__":
    sys.exit(main())
