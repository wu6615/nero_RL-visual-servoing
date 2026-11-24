"""IK accuracy and the delta-pose action convention."""
import numpy as np

from nero_vs_rl.utils.geometry import logmap


def test_home_pose_reached_and_away_from_limits(env):
    ik = env.controller.ik
    p, R = env._home_tcp_target()
    p_fk, R_fk = ik.fk(env.q_home)
    assert np.linalg.norm(p_fk - p) < 2e-3 and np.linalg.norm(logmap(R_fk.T @ R)) < 2e-2
    margin = np.minimum(env.q_home - ik.lo, ik.hi - env.q_home) / (ik.hi - ik.lo)
    assert margin.min() > 0.1


def test_ee_frame_translation_action_moves_along_tcp_axis(env):
    env.reset(seed=3)
    p0, R0 = env.agent.tcp_pose(env.data)
    a = np.zeros(7)
    a[2], a[6] = 1.0, 1.0                                    # 沿 TCP +z（接近方向，朝下）
    for _ in range(3):
        env.step(a)
    p1, _ = env.agent.tcp_pose(env.data)
    d = R0.T @ (p1 - p0)
    assert d[2] > 0.03 and abs(d[0]) < 0.01 and abs(d[1]) < 0.01


def test_workspace_clip_keeps_tcp_above_belt(env):
    env.reset(seed=4)
    a = np.zeros(7)
    a[2], a[6] = 1.0, 1.0
    for _ in range(40):
        env.step(a)
    p, _ = env.agent.tcp_pose(env.data)
    assert p[2] > env.cfg.belt_surface_z + env.cfg.workspace_z_above_belt[0] - 0.01


def test_per_joint_speed_limits_and_leash_scale_with_step():
    """7 per-joint speed limits are honoured per step; the command leash grows with the max step (5 cm/step)."""
    import numpy as np

    from nero_vs_rl.envs import DynamicGraspEnv
    from nero_vs_rl.utils.config import EnvConfig
    lim = [3.1416, 3.1416, 3.1416, 3.927, 3.927, 3.927, 3.927]
    env = DynamicGraspEnv(EnvConfig.from_dict({"max_delta_pos_m": 0.05, "max_joint_speed_rad_s": lim}))
    env.reset(seed=0)
    assert np.isclose(env.controller.leash_m, 0.075)
    q_prev = env.controller.q_cmd.copy()
    for _ in range(3):
        q_tgt, _ = env.controller.compute(env.data, np.r_[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert np.all(np.abs(q_tgt - q_prev) <= np.asarray(lim) * env.control_dt + 1e-9)
        q_prev = q_tgt
    env.close()
