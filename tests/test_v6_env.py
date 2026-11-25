"""v6 env changes (aligned with deployment): full-top visibility, shared target correction, hold-still reward,
objects anywhere across the belt, finger-on-rail collision, success-based model selection."""
from types import SimpleNamespace

import numpy as np
import pytest

from nero_vs_rl import TRAIN_CONFIG
from nero_vs_rl.envs.dynamic_grasp import DynamicGraspEnv
from nero_vs_rl.utils import traj_utils
from nero_vs_rl.utils.config import load_config


@pytest.fixture(scope="module")
def site_cfg():
    return load_config(TRAIN_CONFIG)[0]


def test_site_config_turns_on_v6_options(site_cfg):
    c = site_cfg
    assert c.grasp_obs.visible_rule == "full_top" and c.finger_clearance_m == 0.016     # v7：对齐真机抓取档
    assert c.joint_limit_margin_rad == 0.03 and c.workspace_pullback and c.finger_rail_collision
    assert c.trajectory.lateral_full_belt and c.reward.hold_still < 0
    assert (c.init_tcp_offset_x, c.init_tcp_height) == (-0.08, 0.28)


def test_objects_come_anywhere_across_belt_but_inside_rails(site_cfg):
    c = site_cfg
    env = DynamicGraspEnv(c)
    offs = []
    try:
        for s in range(300):
            env.reset(seed=s)
            size, x, yaw = env.model.geom_size[env.obj["geom"]], env.traj.xy[0, 0], env.traj.yaw[0]
            ext = abs(size[0] * np.cos(yaw)) + abs(size[1] * np.sin(yaw)) if env.obj_type == "box" else size[0]
            assert abs(x - c.belt_center_x) + ext <= c.belt_half_width - traj_utils.RAIL_MARGIN_M + 1e-9
            offs.append(x - c.belt_center_x)
    finally:
        env.close()
    offs = np.array(offs)
    assert offs.min() < -0.11 and offs.max() > 0.11            # 两侧都有贴近护栏的
    assert np.mean(np.abs(offs) > 0.07) > 0.3                   # 不再只走中间 ±7 cm


def test_full_top_rule_is_stricter_than_point_rule(make_env):
    """Same initial state: full_top visible ⇒ point visible, and some half-visible objects only count under point."""
    a = make_env(grasp_obs={"visible_rule": "point"}, trajectory={"lateral_full_belt": True})
    b = make_env(grasp_obs={"visible_rule": "full_top"}, trajectory={"lateral_full_belt": True})
    only_point = 0
    for s in range(150):
        va, vb = a.reset(seed=s)[1]["visible"], b.reset(seed=s)[1]["visible"]
        assert not (vb and not va)
        only_point += va and not vb
    assert only_point > 0


def test_env_controller_keeps_gripper_above_belt(site_cfg):
    """Policy pushing straight down with an open, tilted gripper: commanded gripper never goes below belt + 3 mm."""
    env = DynamicGraspEnv(site_cfg)
    try:
        env.reset(seed=3)
        floor = site_cfg.belt_surface_z + site_cfg.finger_clearance_m
        a = np.array([0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 1.0])     # TCP 系 +z = 向下；同时绕 x 倾斜；夹爪全开
        fixed = 0
        for _ in range(20):
            env.step(a)
            ctl = env.controller
            assert ctl.gripper_lowest_z(ctl.q_cmd, site_cfg.gripper_max_q) >= floor - 1e-3
            fixed += any("抬高" in f for f in ctl.last_fixes)
        assert fixed > 0
    finally:
        env.close()


def test_hold_still_after_success(site_cfg):
    env = DynamicGraspEnv(site_cfg)
    try:
        w = site_cfg.reward
        moving = env.compute_dense_reward(dict(success=True, robot_static=False))
        still = env.compute_dense_reward(dict(success=True, robot_static=True))
        assert still == pytest.approx(w.success * w.scale)
        assert moving == pytest.approx((w.success + w.hold_still) * w.scale) and moving < still
    finally:
        env.close()


def test_finger_on_rail_counts_as_collision_only_when_enabled(site_cfg):
    env = DynamicGraspEnv(site_cfg)
    try:
        env.reset(seed=0)
        lf, rail = env.agent.finger_geoms[0], env.rail_geoms[0]
        data = SimpleNamespace(ncon=1, contact=SimpleNamespace(geom1=np.array([lf]), geom2=np.array([rail])))
        obj = env.obj["geom"]
        assert env.agent.scan_contacts(data, obj, env.obstacle_geoms, env.rail_geoms)["collision"]
        assert not env.agent.scan_contacts(data, obj, env.obstacle_geoms)["collision"]      # v5：手指碰护栏不算
        data.contact.geom2 = np.array([env.belt_geom])
        assert not env.agent.scan_contacts(data, obj, env.obstacle_geoms, env.rail_geoms)["collision"]
    finally:
        env.close()


def test_success_eval_callback_uses_fixed_episodes_and_saves_best(tmp_path):
    from stable_baselines3 import SAC
    from nero_vs_rl.algorithms.Networks import GraspPointGroupExtractor
    from nero_vs_rl.algorithms.rl_utils import SuccessEvalCallback
    from nero_vs_rl.utils.config import EnvConfig
    cfg = EnvConfig.from_dict({"max_steps": 5})
    env = DynamicGraspEnv(cfg)
    from stable_baselines3.common.logger import configure
    model = SAC("MultiInputPolicy", env, buffer_size=100, learning_starts=10, device="cpu",
                policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=[16]))
    model.set_logger(configure(str(tmp_path / "log"), ["csv"]))       # 训练外单独调用时要手动给 logger
    cb = SuccessEvalCallback(cfg, eval_freq=1, n_episodes=3, run_dir=tmp_path)
    cb.init_callback(model)
    cb.evaluate()
    first = np.load(tmp_path / "eval" / "evaluations.npz")["belt_speeds"][0].copy()
    cb.evaluate()
    z = np.load(tmp_path / "eval" / "evaluations.npz")
    assert z["successes"].shape == (2, 3) and np.allclose(z["belt_speeds"][1], first)    # 两次评估是同一批回合
    assert (tmp_path / "best" / "best_model.zip").exists()
    env.close()


# ---------------------------------------------------------------- v7
def test_v7_static_and_slow_objects(site_cfg):
    from nero_vs_rl.utils.config import EnvConfig
    rng = np.random.default_rng(0)
    speeds = []
    for _ in range(600):
        tr = traj_utils.generate(rng, site_cfg, 3.0)
        speeds.append(float(np.linalg.norm(tr.vel, axis=1).max()))
    v = np.array(speeds)
    assert 0.06 < np.mean(v == 0.0) < 0.14                                   # 约 10% 静止
    slow = np.mean((v > 0) & (v <= site_cfg.trajectory.slow_speed[1] + 1e-9))
    assert 0.17 < slow < 0.30                                                # 90% × 25% ≈ 22% 慢带
    assert v.max() <= site_cfg.trajectory.speed[1] + 1e-9
    old = EnvConfig()                                                        # 旧配置不多抽随机数：同种子轨迹不变
    a, b = traj_utils.generate(np.random.default_rng(5), old, 2.0), traj_utils.generate(np.random.default_rng(5), old, 2.0)
    assert np.allclose(a.xy, b.xy) and old.trajectory.static_prob == 0 and old.trajectory.slow_prob == 0


def test_v7_finger_length_noise_moves_fingertips_and_twin_stays_nominal(site_cfg):
    from nero_vs_rl.sim2real.deploy.common import RobotTwin
    env = DynamicGraspEnv(site_cfg)
    try:
        offs, tips = [], []
        q = None
        for s in range(30):
            env.reset(seed=s)
            q = env.q_home if q is None else q
            P = env.controller.gripper_points(q, site_cfg.gripper_max_q)
            tips.append(P[:, 2].min()); offs.append(env.finger_length_offset)
        offs, tips = np.array(offs), np.array(tips)
        assert np.abs(offs).max() <= site_cfg.finger_length_noise_m + 1e-12 and offs.std() > 0.001
        assert np.corrcoef(offs, tips)[0, 1] < -0.95                         # 手指越长，指尖越低（夹爪朝下）
        twin = RobotTwin(site_cfg)
        assert twin.cfg.finger_length_noise_m == 0.0
    finally:
        env.close()


def test_fingers_block_the_wrist_camera(site_cfg):
    """mj_ray skips transparent geoms: the collision meshes must stay opaque or finger occlusion silently does nothing."""
    import dataclasses

    import mujoco

    from nero_vs_rl.sensors import camera as cam
    env = DynamicGraspEnv(dataclasses.replace(site_cfg, finger_length_noise_m=0.0))
    try:
        env.reset(seed=0)
        m, d, c = env.model, env.data, site_cfg.camera
        assert all(m.geom_rgba[g, 3] > 0 for g in env.agent.finger_geoms)
        o, gid = d.cam_xpos[env.agent.camera].copy(), np.zeros(1, np.int32)
        for g in env.agent.finger_geoms:
            v = d.geom_xpos[g] - o
            mujoco.mj_ray(m, d, o, v / np.linalg.norm(v), cam.RAY_GEOMGROUP, 1, -1, gid)
            assert gid[0] in env.agent.finger_geoms
        x, y = np.meshgrid(np.linspace(-.13, .13, 30), np.linspace(-.2, .3, 40))
        pts = np.column_stack([site_cfg.belt_center_x + x.ravel(), y.ravel(), np.full(x.size, site_cfg.belt_surface_z + 0.05)])
        free = cam.visible_mask(m, d, env.agent.camera, pts, -1, dataclasses.replace(c, occlusion_check=False))
        seen = cam.visible_mask(m, d, env.agent.camera, pts, -1, c)
        assert not (seen & ~free).any() and 0.03 < (free & ~seen).sum() / free.sum() < 0.4      # 手指挡住一部分，不是全部
    finally:
        env.close()


def test_visibility_agrees_with_deployed_perception(site_cfg):
    """The simulated "seen" flag must track what perception.py accepts on the rendered depth (start states)."""
    import dataclasses

    from nero_vs_rl.sim2real.deploy.common import RobotTwin
    from nero_vs_rl.sim2real.deploy.perception import BeltObjectDetector, observe
    from nero_vs_rl.sim2real.hardware.sim_workcell import sim_depth_intrinsics
    from nero_vs_rl.utils.geometry import make_T
    cfg = dataclasses.replace(site_cfg, finger_length_noise_m=0.0)
    env, tw, i = DynamicGraspEnv(cfg), RobotTwin(cfg), sim_depth_intrinsics(cfg)
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
    same, n = 0, 120
    try:
        for s in range(n):
            vis = env.reset(seed=s)[1]["visible"]
            depth = env.render_wrist_rgbd()[1]
            tw.set_joints(env.agent.arm_qpos(env.data))
            p, R = tw.tcp_pose()
            G = observe(BeltObjectDetector(cfg, K, np.zeros(5), tw.max_width), depth, tw.T_base_cam_cv(), make_T(R, p), *tw.cam_pose_mj(), cfg)[1]
            same += vis == (G is not None)
    finally:
        env.close()
    assert same / n > 0.92
