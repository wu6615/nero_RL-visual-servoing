"""Step 2 (server) - joint/body names, limits and body poses must match our kinematics table; contact sensors must work.

  python -m nero_vs_rl.isaac.scripts.step2_check_assets --headless --num-envs 16
"""
from __future__ import annotations

import argparse
import sys

from ._app import finish, start

# =============================================================================
# 【说明】验收标准：随机关节角下 gripper_base 位姿和运动学表差 < 0.5 mm / 0.1°；关节限位差 < 1e-3；
#   物体尺寸规格、质量和环境一一对应；物体落在带面上时不穿透、跟着带走；
#   把物体放到两指之间时，物体的接触传感器能报出左右手指。
# =============================================================================


def main():
    args, app, tc, ic = start(argparse.ArgumentParser(description="核对 USD、运动学和接触传感器"))
    code = 1
    try:
        import torch
        from ..core.so3 import quat_wxyz_to_mat, rot_angle
        from ..sim import scene as S
        from ..sim.grasp_env import make_env
        env = make_env(tc, ic, args.num_envs or 16, args.device, args.seed)
        ok = True

        def item(good, text):
            nonlocal ok
            ok &= bool(good)
            print(f"  {'✓' if good else '✗'} {text}")

        env.reset()
        r, kin, B = env.robot, env.task.kin, env.num_envs
        item(set(S.BODIES) <= set(r.body_names), f"body 名：{r.body_names}")
        lim = r.data.joint_pos_limits[0, env.arm_ids]
        item(torch.allclose(lim[:, 0], kin.lo, atol=1e-3) and torch.allclose(lim[:, 1], kin.hi, atol=1e-3), "关节限位和模型一致")
        gb = r.body_names.index("gripper_base")
        worst_p = worst_r = 0.0
        for _ in range(64):                                                        # 64 × num_envs 组随机关节角
            q = kin.lo + (kin.hi - kin.lo) * (0.1 + 0.8 * torch.rand(B, 7, device=env.device))
            r.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=env.arm_ids)
            r.set_joint_position_target(q, joint_ids=env.arm_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            r.update(0.0)
            R, p = kin.gripper_base(q)
            dp = (r.data.body_link_pos_w[:, gb] - env.scene.env_origins - p).norm(dim=1).max()
            dr = rot_angle(R.transpose(1, 2) @ quat_wxyz_to_mat(r.data.body_link_quat_w[:, gb])).max()
            worst_p, worst_r = max(worst_p, float(dp)), max(worst_r, float(dr))
        item(worst_p < 5e-4 and worst_r < 0.0018, f"gripper_base 位姿最大差 {worst_p * 1000:.2f} mm / {worst_r * 57.3:.3f}°")
        mass = env.object.root_physx_view.get_masses().reshape(-1).to(env.device)
        want = torch.tensor([env.task.specs[i % len(env.task.specs)].mass for i in range(B)], device=env.device)
        item(torch.allclose(mass, want, atol=1e-4), "物体规格和环境一一对应（按质量核对）")

        env.reset()
        env.task.attached[:] = False                                               # 放开驱动，只靠带面摩擦
        hold = torch.zeros(B, 7, device=env.device)
        hold[:, 6] = 1.0
        y0 = env._state().obj_pos[:, 1].clone()
        t0 = env.task.traj.y.clone()
        for _ in range(8):
            env.step(hold)
        s = env._state()
        dz = (s.obj_pos[:, 2] - env.task.obj.half[:, 2] - tc.belt_surface_z).abs().max()
        slip = ((s.obj_pos[:, 1] - y0) - (env.task.traj.y - t0)).abs().max()
        item(dz < 2e-3, f"物体停在带面上（高度差最大 {dz * 1000:.1f} mm）")
        item(slip < 0.02, f"物体被带面带走（0.8 s 内和带面的位移差最大 {slip * 100:.1f} cm）")

        # 一指在物体左、一指在右：物体放到指尖处再闭合
        env.reset()
        env.task.attached[:] = False
        Rg, pg = kin.gripper_base(env._state().q)
        tip = pg + Rg @ (kin.p_tcp + torch.tensor([0.0, 0.0, 0.005], device=env.device))
        seen = torch.zeros(B, dtype=torch.bool, device=env.device)
        close = torch.zeros(B, 7, device=env.device)
        close[:, 6] = -1.0
        yaw = torch.atan2(Rg[:, 1, 0], Rg[:, 0, 0])
        for _ in range(12):
            env._place_object(torch.arange(B, device=env.device), tip, yaw, torch.zeros(B, 3, device=env.device))
            env.step(close)
            seen |= env.left & env.right
        item(seen.float().mean() > 0.8, f"两指接触传感器：{int(seen.sum())}/{B} 个环境同时报出左右手指")
        print("全部通过" if ok else "有没通过的项，先解决再往下走")
        code = 0 if ok else 1
        env.close()
    finally:
        finish(app, code)


if __name__ == "__main__":
    sys.exit(main())
