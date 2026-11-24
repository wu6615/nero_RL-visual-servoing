"""Env contract: spaces, determinism, perception model, termination, scripted solvability."""
import numpy as np
import pytest

from nero_vs_rl.algorithms.scripted_pbvs import ScriptedGraspPolicy
from nero_vs_rl.sensors import camera as cam


def test_reset_step_respect_spaces(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)


def test_reset_is_deterministic_given_seed(make_env):
    e = make_env()
    o1, _ = e.reset(seed=11)
    p1 = e.data.xpos[e.obj["body"]].copy()
    e.reset(seed=99)
    o2, _ = e.reset(seed=11)
    assert np.allclose(p1, e.data.xpos[e.obj["body"]])
    assert np.allclose(o1["tcp_pose"], o2["tcp_pose"])


def test_padded_grasp_slots_are_zero(env):
    obs, info = env.reset(seed=0)
    valid = obs["grasp_scores"] > 0
    assert np.all(obs["gripper_pts_diff"][~valid] == 0)
    assert np.array_equal(valid, info["observed_valid"])


def test_blind_zone_hides_points_too_close():
    cfg = cam.CameraConfig(min_depth_m=0.15, max_depth_m=1.2)
    cam_pos, cam_mat = np.zeros(3), np.eye(3)               # 视线 -Z
    pts = np.array([[0, 0, -0.10], [0, 0, -0.50], [0, 0, 0.5], [2.0, 0, -0.5]])
    uv, depth = cam.project(pts, cam_pos, cam_mat, cfg)
    inside = (depth >= cfg.min_depth_m) & (depth <= cfg.max_depth_m) & (uv[:, 0] >= 0) & (uv[:, 0] <= cfg.width)
    assert inside.tolist() == [False, True, False, False]


def test_target_is_visible_at_start_and_memory_extrapolates(make_env):
    e = make_env(grasp_obs={"memory_mode": "extrapolate"})
    obs, info = e.reset(seed=0)
    assert info["visible"]
    # 把相机视场角压到极小 → 看不见，但记忆仍提供抓取，并按速度外推
    e.model.cam_fovy[e.agent.camera] = 0.01
    e.cfg.camera.fovy_deg = 0.01
    hold = np.zeros(7)
    hold[6] = 1
    obs, _, _, _, info = e.step(hold)
    assert not info["visible"] and info["observed_valid"].any() and obs["grasp_vis"][1] > 0


def test_object_leaving_downstream_ends_episode(make_env):
    e = make_env(trajectory={"speed": [0.30, 0.30], "vary_speed": False})
    e.reset(seed=1)
    hold = np.zeros(7)
    hold[6] = 1
    for _ in range(e.cfg.max_steps):
        _, _, term, trunc, info = e.step(hold)
        if term or trunc:
            break
    assert term and info["missed"] and not info["success"]


def test_low_rails_on_both_sides_of_belt(env):
    """Site: low rails (~1.5 cm) on both belt edges, static, outside the belt surface, counted as obstacles."""
    c, m = env.cfg, env.model
    xs = []
    for g in env.rail_geoms:
        pos, size = m.geom_pos[g], m.geom_size[g]
        assert m.geom_bodyid[g] == 0                                   # 世界系静止，不随带面滑动
        assert pos[2] + size[2] == pytest.approx(c.belt_surface_z + c.belt_rail_height_m)
        assert abs(pos[0] - c.belt_center_x) - size[0] > c.belt_half_width   # 不压在带面上
        assert g in env.obstacle_geoms
        xs.append(pos[0] - c.belt_center_x)
    assert min(xs) < 0 < max(xs)                                       # 两侧各一条


@pytest.mark.parametrize("obj", ["box", "cylinder"])
def test_dropped_object_is_carried_by_belt(make_env, obj):
    """A target that slips out of the gripper and lands on the belt must move on with the belt, then leave downstream."""
    v = 0.20
    e = make_env(trajectory={"speed": [v, v], "vary_speed": False})
    e.reset(seed=3, options={"object_types": [obj]})
    hold = np.zeros(7)
    hold[6] = 1
    e.step(hold)
    o, d = e.obj, e.data
    d.eq_active[o["eq"]] = 0                                  # 模拟抓起后松脱：脱离驱动，从 4 cm 高处静止落下
    e._attached = False
    d.qpos[o["qadr"] + 2] += 0.04
    d.qvel[o["dadr"]:o["dadr"] + 6] = 0
    ys = []
    for _ in range(12):
        _, _, term, trunc, info = e.step(hold)
        ys.append(d.xpos[o["body"]][1])
    assert not info["fell"]
    v_obj = (ys[-1] - ys[5]) / (6 * e.control_dt)            # 落地、加速后的平均速度
    assert v_obj == pytest.approx(-v, rel=0.15)
    for _ in range(e.cfg.max_steps):
        if term or trunc:
            break
        _, _, term, trunc, info = e.step(hold)
    assert info["missed"]                                     # 掉回带上的物体照样会流出下游界限


@pytest.mark.parametrize("mode,min_success", [("static", 0.5), ("line", 0.5)])
def test_scripted_pbvs_solves_task(make_env, mode, min_success):
    e = make_env(trajectory={"mode": mode})
    pol = ScriptedGraspPolicy(e.cfg)
    wins = unstable = 0
    for ep in range(10):
        obs, info = e.reset(seed=100 + ep)
        pol.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = e.step(pol(obs, info))
            done = term or trunc
        wins += info["success"]
        unstable += info.get("sim_unstable", False)
    assert unstable == 0
    assert wins / 10 >= min_success


def test_time_penalty_and_early_failure_compensation(make_env):
    """Each pre-success step costs reward.time; ending early by missed/fell charges the remaining steps too."""
    e = make_env(max_steps=80)
    e.reset(seed=0)
    w = e.cfg.reward
    base = dict(success=False, visible=False, dist_t=0.5, dist_r=0.5, grasped=False, lifted=False, lift=0.0,
                robot_static=True, collision=False, missed=False, fell=False)
    e.step_count = 10
    no_time = e.compute_dense_reward(base) - w.time * w.scale
    assert np.isclose(e.compute_dense_reward(base), no_time + w.time * w.scale)
    missed = e.compute_dense_reward(dict(base, missed=True))
    assert np.isclose(missed, no_time + w.time * w.scale * (1 + 80 - 10))   # 剩余 70 步一次扣掉
    assert e.compute_dense_reward(dict(base, success=True)) == w.success * w.scale   # 成功后不再扣时间
    assert w.time < 0
