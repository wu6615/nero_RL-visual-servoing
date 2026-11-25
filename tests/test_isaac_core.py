"""Isaac batch core vs the MuJoCo/numpy versions (no Isaac needed; only kinematics, no physics stepping)."""
import subprocess
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from nero_vs_rl import TRAIN_CONFIG
from nero_vs_rl.envs.dynamic_grasp import DynamicGraspEnv
from nero_vs_rl.envs.observation import GraspTracker, assemble_obs as np_obs
from nero_vs_rl.grasp_detection import analytic_grasps as gr
from nero_vs_rl.isaac import ROBOT_URDF
from nero_vs_rl.isaac.config import load_isaac_config
from nero_vs_rl.isaac.core import reward as R_
from nero_vs_rl.isaac.core.controller import solve_home
from nero_vs_rl.isaac.core.kinematics import NeroKinematics
from nero_vs_rl.isaac.core.objects import BOX, ObjectBatch, make_specs
from nero_vs_rl.isaac.core.observation import action_space, assemble_obs, observation_space
from nero_vs_rl.isaac.core.perception import GraspMemory, WristCamera
from nero_vs_rl.isaac.core.so3 import expmap, logmap, quat_wxyz_to_mat, symmetric_rot_dist
from nero_vs_rl.isaac.core.task import GraspTask, SimState
from nero_vs_rl.sensors import camera as cam
from nero_vs_rl.utils import geometry as G
from nero_vs_rl.utils.config import load_config

F64 = torch.float64
t = lambda x: torch.as_tensor(np.asarray(x), dtype=F64)


@pytest.fixture(scope="module")
def cfg():
    return load_config(TRAIN_CONFIG)[0]


@pytest.fixture(scope="module")
def env(cfg):
    import dataclasses
    e = DynamicGraspEnv(dataclasses.replace(cfg, finger_length_noise_m=0.0))
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def kin():
    return NeroKinematics(dtype=F64)


def _rand_q(kin, n, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = kin.lo.numpy(), kin.hi.numpy()
    return lo + (hi - lo) * rng.uniform(0.05, 0.95, (n, 7))


# ---------------------------------------------------------------- geometry
def test_so3_matches_numpy():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1.2, (300, 3))
    w[:20] *= 1e-8
    w[20:40] = w[20:40] / np.linalg.norm(w[20:40], axis=1, keepdims=True) * (np.pi - 1e-6)
    E = expmap(t(w)).numpy()
    assert np.allclose(E, [G.expmap(x) for x in w], atol=1e-9)
    back = expmap(logmap(t(E))).numpy()
    assert np.allclose(back, E, atol=1e-5)                          # 含接近 180° 的
    assert np.allclose(symmetric_rot_dist(t(E)).numpy(), G.symmetric_grasp_rot_dist(E), atol=1e-6)
    q = rng.normal(size=(50, 4))
    assert np.allclose(quat_wxyz_to_mat(t(q)).numpy(), [G.quat_wxyz_to_mat(x) for x in q])


# ---------------------------------------------------------------- kinematics
def test_kinematics_jacobian_camera_and_gripper_height(env, kin, cfg):
    m, d, ctl = env.model, mujoco.MjData(env.model), env.controller
    camera = WristCamera(kin, cfg)
    qs, grip = _rand_q(kin, 60), np.random.default_rng(1).uniform(0, 0.05, 60)
    p, R, J = kin.tcp_jacobian(t(qs))
    Rg, pg = kin.gripper_base(t(qs))
    cp, cR = camera.pose(Rg, pg)
    low = kin.gripper_lowest_z(t(qs), t(grip))
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for i, q in enumerate(qs):
        env.agent.set_state(d, q, grip[i])
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_camlight(m, d)
        pm, Rm = env.agent.tcp_pose(d)
        mujoco.mj_jacSite(m, d, jp, jr, env.agent.tcp_site)
        assert np.allclose(p[i], pm, atol=1e-8) and np.allclose(R[i], Rm, atol=1e-8)
        assert np.allclose(J[i], np.vstack([jp, jr])[:, ctl.ik.dadr], atol=1e-8)
        assert np.allclose(cp[i], d.cam_xpos[env.agent.camera], atol=1e-8)
        assert np.allclose(cR[i], d.cam_xmat[env.agent.camera].reshape(3, 3), atol=1e-7)
        assert abs(float(low[i]) - ctl.gripper_lowest_z(q, grip[i])) < 1e-7


def test_urdf_chain_matches_table(kin):
    """The URDF handed to Isaac describes the same arm."""
    joints = {j.find("child").get("link"): j for j in ET.parse(ROBOT_URDF).getroot().findall("joint")}
    q = _rand_q(kin, 1, 3)[0]
    T = np.eye(4)
    for name in [f"link{i}" for i in range(1, 8)] + ["gripper_flange", "gripper_base"]:
        j = joints[name]
        o = j.find("origin")
        A = np.eye(4)
        A[:3, :3] = Rotation.from_euler("xyz", [float(x) for x in o.get("rpy").split()]).as_matrix()
        A[:3, 3] = [float(x) for x in o.get("xyz").split()]
        T = T @ A
        if j.get("type") == "revolute":
            B = np.eye(4)
            B[:3, :3] = G.rot_z(q[int(name[-1]) - 1])
            T = T @ B
    Rg, pg = kin.gripper_base(t(q[None]))
    assert np.allclose(T[:3, 3], pg[0], atol=1e-7) and np.allclose(T[:3, :3], Rg[0], atol=1e-7)
    assert {j.get("name") for j in joints.values() if j.get("type") == "prismatic"} == {"gripper_joint1", "gripper_joint2"}


def test_home_and_controller_match_numpy(env, kin, cfg):
    assert np.allclose(solve_home(kin, cfg).numpy(), env.q_home, atol=1e-4)
    task = GraspTask(cfg, 1, 4, 0, dtype=F64)
    d = mujoco.MjData(env.model)
    rng = np.random.default_rng(5)
    worst = 0.0
    for trial in range(25):
        q = env.q_home + rng.normal(0, 0.12, 7)
        g = rng.uniform(0, 0.05)
        env.agent.set_state(d, q, g)
        mujoco.mj_forward(env.model, d)
        env.controller.reset(d)
        task.ctl.reset(torch.tensor([0]), t(q[None]))
        for _ in range(6):                                       # 连续 6 步，带累积指令；含往下压（触发指尖抬高）
            a = rng.uniform(-1, 1, 7)
            a[2] = abs(a[2]) if trial % 2 else a[2]
            qn, gn = env.controller.compute(d, a)
            qt, gt = task.ctl.compute(t(q[None]), t([g]), t(a[None]))
            worst = max(worst, float(np.abs(qt[0].numpy() - qn).max()))
            assert abs(float(gt[0]) - gn) < 1e-9
            q = q + 0.7 * (qn - q)                               # 假的跟随：只动运动学，不跑物理
            env.agent.set_state(d, q, g)
            mujoco.mj_forward(env.model, d)
    assert worst < 1e-3, worst


# ---------------------------------------------------------------- perception / observation
def test_visibility_projection_exact_and_occlusion_close(env, kin, cfg):
    import dataclasses
    rng = np.random.default_rng(2)
    d = mujoco.MjData(env.model)
    plain = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera, occlusion_check=False))
    c0, c1 = WristCamera(kin, plain), WristCamera(kin, cfg)
    agree = n = blocked = 0
    for _ in range(40):
        q, g = env.q_home + rng.normal(0, 0.15, 7), rng.uniform(0.0, 0.05)
        env.agent.set_state(d, q, g)
        mujoco.mj_forward(env.model, d)
        p, R = env.agent.tcp_pose(d)
        # 一半点在带面附近（训练时的情况），一半在"相机 → 手指表面"射线的延长线附近（专门制造遮挡）
        far = np.column_stack([p[0] + rng.uniform(-.2, .2, 30), p[1] + rng.uniform(-.2, .2, 30), rng.uniform(-.10, -.02, 30)])
        Rg, pg = kin.gripper_base(t(q[None]))
        o = c0.pose(Rg, pg)[0][0].numpy()
        v = np.concatenate([(kin.verts_f1 + kin.slide1 * g).numpy(), (kin.verts_f2 + kin.slide2 * g).numpy()])
        v = v[rng.integers(0, len(v), 30)] @ Rg[0].numpy().T + pg[0].numpy() + rng.normal(0, 0.006, (30, 3))
        near = o + (v - o) * rng.uniform(1.8, 3.0, (30, 1))
        pts = np.concatenate([far, near])
        Rg, pg = kin.gripper_base(t(q[None]))
        a = c0.visible(t(pts[None]), Rg, pg, t([g]))[0].numpy()
        ref = cam.visible_mask(env.model, d, env.agent.camera, pts, -1, plain.camera)
        assert (a == ref).all()
        b = c1.visible(t(pts[None]), Rg, pg, t([g]))[0].numpy()
        ref = cam.visible_mask(env.model, d, env.agent.camera, pts, env.obj["geom"], cfg.camera)
        agree, n, blocked = agree + (b == ref).sum(), n + a.sum(), blocked + (a & ~ref).sum()
        agree -= (~a & ~ref).sum()                               # 只统计投影可见的点
    print(f"遮挡判断一致 {agree}/{n}，真被挡 {blocked}")
    assert blocked > 20 and agree / n > 0.95


def test_seen_or_not_matches_mujoco_env(cfg):
    """Whole visibility rule (frame, depth, share of the top exposed) on the same start states."""
    import dataclasses
    e = DynamicGraspEnv(dataclasses.replace(cfg, finger_length_noise_m=0.0))
    task = GraspTask(cfg, 1, 1, 0, dtype=F64)
    same = seen = 0
    try:
        for s in range(150):
            info = e.reset(seed=s)[1]
            half = e.model.geom_size[e.obj["geom"]].copy()
            spec = task.specs[0]
            spec.kind, spec.half = (BOX, half) if e.obj_type == "box" else (1, half[[0, 0, 1]])
            task.obj = ObjectBatch([spec], cfg, 1, dtype=F64)
            task.mem = GraspMemory(cfg, 1, task.obj.G, "cpu", F64)
            b, f = e.obj["body"], torch.tensor([False])
            st = SimState(t(e.agent.arm_qpos(e.data)[None]), t(np.zeros((1, 7))), t([e.agent.gripper_opening(e.data)]),
                          t(e.data.xpos[b][None]), t(e.data.xquat[b][None]), t(np.zeros((1, 3))), f, f, f, f)
            task.observe(st, task.all_ids)
            same, seen = same + (bool(task.visible[0]) == info["visible"]), seen + info["visible"]
    finally:
        e.close()
    print(f"看到与否一致 {same}/150，看到 {seen}")
    assert same >= 145 and 60 < seen < 140


def test_memory_and_observation_match_numpy(cfg):
    g, dt, M = cfg.grasp_obs, 0.1, 18
    rng = np.random.default_rng(3)
    tr, mem = GraspTracker(g, dt), GraspMemory(cfg, 1, M, "cpu", F64)
    ids = torch.tensor([0])
    for k in range(80):
        T = np.tile(np.eye(4), (M, 1, 1))
        for i in range(M):
            T[i, :3, :3], T[i, :3, 3] = G.expmap(rng.normal(size=3)), rng.normal(0, 0.2, 3)
        sc, vis = rng.uniform(0.5, 1, M), rng.random(M) < 0.6
        seen = not (20 < k < 60 and k % 7) and k != 0                # 第一帧没看到；中间长时间看不到（超过记忆年龄）
        pos = rng.normal(size=3)
        vis = vis & seen
        Gw, s, v = tr.update(k * dt, T[vis] if vis.any() else None, sc[vis], pos)
        Tt, st, use, age = mem.update(ids, t([k * dt]), t(T[None]), t(sc[None]), torch.tensor(vis[None]), torch.tensor([bool(vis.any())]), t(pos[None]))
        assert np.allclose(mem.vel[0], tr.est_obj_vel, atol=1e-8) and abs(float(age[0]) - tr.age_norm) < 1e-12
        p, R, pts = rng.normal(size=3), G.expmap(rng.normal(size=3)), rng.normal(0, .02, (g.num_points, 3))
        pa, gp = rng.uniform(-1, 1, 7), rng.uniform(0, 1)
        a, _, _ = np_obs(p, R, Gw, s, v, tr.age_norm, gp, pa, pts, g.num_grasps)
        b = assemble_obs(t(p[None]), t(R[None]), Tt, st, use, torch.tensor([v]), age, t([gp]), t(pa[None]), t(pts[None]), g.num_grasps)
        for key in a:
            assert np.allclose(a[key], b[key][0].numpy(), atol=1e-5), (k, key)


def test_spaces_equal_mujoco_env(env, cfg):
    assert observation_space(cfg.grasp_obs.num_grasps, cfg.grasp_obs.num_points) == env.observation_space
    assert action_space() == env.action_space


# ---------------------------------------------------------------- objects / reward
def test_objects_use_the_same_grasps_and_jaw_fit(cfg):
    specs = make_specs(cfg, 10, 0)
    b = ObjectBatch(specs, cfg, 23, dtype=F64)
    max_w = 0.008 + 2 * cfg.gripper_max_q - cfg.grasp_obs.finger_margin_m
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(23, 3))
    mis, wid = b.jaw_fit(t(dirs))
    for i in range(23):
        s = specs[i % 10]
        ref = gr.box_grasps(s.half, max_w, 15.0) if s.kind == BOX else gr.cylinder_grasps(s.half[0], s.half[2], max_w, 15.0)
        assert int(b.valid[i].sum()) == len(ref) and np.allclose(b.T_obj[i, :len(ref)], ref.T_obj)
        half = s.half if s.kind == BOX else [s.half[0], s.half[2], 0]
        a, w = gr.jaw_fit("box" if s.kind == BOX else "cylinder", np.asarray(half), dirs[i])
        assert abs(float(mis[i]) - a) < 1e-6 and abs(float(wid[i]) - w) < 1e-9


def test_reward_matches_numpy(cfg):
    rng = np.random.default_rng(0)
    for _ in range(300):
        B = {k: bool(rng.random() < 0.4) for k in ("grasped", "lifted", "robot_static", "collision", "visible", "missed", "fell", "closing")}
        f = dict(dist_t=rng.uniform(0, .6), dist_r=rng.uniform(0, 1), lift=rng.uniform(-.05, .2))
        jaw = dict(misalign=rng.uniform(0, 1.5), tilt=rng.uniform(0, 1.5), rel_speed=rng.uniform(0, 1))
        ok, steps = bool(rng.random() < 0.2), int(rng.integers(1, cfg.max_steps + 1))
        e = SimpleNamespace(cfg=cfg, step_count=steps)
        e.closing_penalty = lambda i, e=e: DynamicGraspEnv.closing_penalty(e, i)
        ref = DynamicGraspEnv.compute_dense_reward(e, dict(B, **f, success=ok, jaw=jaw))
        info = {k: torch.tensor([v]) for k, v in dict(B, success=ok).items()} | {k: t([v]) for k, v in (f | jaw).items()}
        assert float(R_.dense_reward(cfg, info, torch.tensor([steps]))[0]) == pytest.approx(ref, abs=1e-6)


def test_evaluate_distances_match_env(env, cfg):
    task = GraspTask(cfg, 1, 1, 0, dtype=F64)
    spec = task.specs[0]
    spec.kind, spec.half = (0, env.model.geom_size[env.targets["box"]["geom"]].copy())
    while env.obj_type != "box":
        env.reset(seed=int(np.random.default_rng(0).integers(1e6)) if env.obj_type == "x" else env.step_count + 1 + getattr(env, "_s", 0))
        env._s = getattr(env, "_s", 0) + 1
    spec.half = env.model.geom_size[env.obj["geom"]].copy()
    task.obj = ObjectBatch([spec], cfg, 1, dtype=F64)
    b = env.obj["body"]
    info = env.evaluate()
    s = SimState(q=t(env.agent.arm_qpos(env.data)[None]), qd=t(np.zeros((1, 7))), grip=t([env.agent.gripper_opening(env.data)]),
                 obj_pos=t(env.data.xpos[b][None]), obj_quat=t(env.data.xquat[b][None]), obj_vel=t(np.zeros((1, 3))),
                 left=torch.tensor([False]), right=torch.tensor([False]), touch=torch.tensor([False]), collision=torch.tensor([False]))
    j = task.evaluate(s)
    for k in ("dist_t", "dist_r", "lift"):
        assert float(j[k][0]) == pytest.approx(info[k], abs=1e-6), k
    for k in ("misalign", "width", "gap", "tilt"):
        assert float(j[k][0]) == pytest.approx(info["jaw"][k], abs=1e-6), k


# ---------------------------------------------------------------- whole task, SB3 adapter, policy transfer
class KinematicSim:
    """Stand-in simulator: joints reach their targets at once, untouched objects follow the belt (no physics)."""

    def __init__(self, cfg, n):
        self.task, self.num_envs, self.device = GraspTask(cfg, n, 8, 0, seed=1), n, "cpu"
        self.q, self.g = torch.zeros(n, 7), torch.full((n,), 0.05)
        self.terminal_obs = self.task_info = None

    def _state(self):
        pos, yaw, vel = self.task.object_drive()
        z, f = torch.zeros(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool)
        quat = torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], 1)
        return SimState(self.q, torch.zeros_like(self.q), self.g, pos, quat, vel, f, f, f, f)

    def reset(self):
        ids = self.task.all_ids
        self.q = self.task.reset(ids)
        return {"policy": self.task.observe(self._state(), ids)}, {}

    def step(self, a):
        _, self.q, self.g = self.task.pre_step(self._state(), a)
        for _ in range(20):
            self.task.substep(0.005)
        obs, rew, term, trunc, self.task_info = self.task.post_step(self._state())
        self.terminal_obs = obs
        out = {k: v.clone() for k, v in obs.items()}
        ids = (term | trunc).nonzero().squeeze(-1)
        if len(ids):
            self.q[ids], self.g[ids] = self.task.reset(ids), 0.05
            for k, v in self.task.observe(self._state(), ids).items():
                out[k][ids] = v
        return {"policy": out}, rew, term, trunc, {}

    def close(self):
        pass


def test_task_runs_and_sb3_adapter_reports_terminal_obs(cfg):
    import dataclasses
    from stable_baselines3 import SAC
    from nero_vs_rl.algorithms.Networks import GraspPointGroupExtractor
    from nero_vs_rl.isaac.sb3_vec_env import IsaacSb3VecEnv
    c = dataclasses.replace(cfg, max_steps=6)
    v = IsaacSb3VecEnv(KinematicSim(c, 5), c.grasp_obs.num_grasps, c.grasp_obs.num_points)
    obs = v.reset()
    assert all(obs[k].shape == (5, *s.shape) for k, s in v.observation_space.spaces.items())
    done_seen = 0
    for k in range(13):
        obs, rew, done, infos = v.step(np.random.default_rng(k).uniform(-1, 1, (5, 7)).astype(np.float32))
        assert np.isfinite(rew).all() and all(np.isfinite(x).all() for x in obs.values())
        for i in np.flatnonzero(done):
            assert infos[i]["TimeLimit.truncated"] and infos[i]["episode"]["l"] == 6 and not infos[i]["is_success"]
            assert not np.allclose(infos[i]["terminal_observation"]["action"], 0) and np.allclose(obs["action"][i], 0)   # 终止观测在重置前
            done_seen += 1
    assert done_seen == 10
    m = SAC("MultiInputPolicy", v, buffer_size=200, batch_size=16, learning_starts=10, train_freq=1, gradient_steps=2, device="cpu",
            policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=[16]))
    m.learn(60)


def test_policy_roundtrip_keeps_actions(tmp_path, cfg):
    from stable_baselines3 import SAC
    from nero_vs_rl.algorithms.Networks import GraspPointGroupExtractor
    from nero_vs_rl.isaac.policy_io import export_policy, import_policy
    from nero_vs_rl.isaac.sb3_vec_env import IsaacSb3VecEnv
    from nero_vs_rl.sim2real.deploy.runtime.session import load_policy
    g = cfg.grasp_obs
    v = IsaacSb3VecEnv(KinematicSim(cfg, 2), g.num_grasps, g.num_points)
    m = SAC("MultiInputPolicy", v, buffer_size=10, device="cpu", seed=3, policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=[32, 32]))
    export_policy(m, tmp_path / "p.pt", [32, 32], g.num_grasps, g.num_points)
    import_policy(tmp_path / "p.pt", tmp_path / "p.zip", cfg)
    obs = {k: x[0] for k, x in v.reset().items()}
    assert np.allclose(load_policy(str(tmp_path / "p.zip"))(obs), m.predict(obs, deterministic=True)[0], atol=1e-6)


def test_trajectory_distribution(cfg):
    task = GraspTask(cfg, 4000, 16, 0, seed=2)
    task.reset(task.all_ids)
    tr, tc = task.traj, cfg.trajectory
    off = (tr.x0 - cfg.belt_center_x).abs() + task.obj.half_width_x(tr.yaw)
    assert float(off.max()) <= cfg.belt_half_width - 0.005 + 1e-5
    static, v = tr.hi == 0, tr.v0
    assert 0.07 < float(static.float().mean()) < 0.13
    assert 0.18 < float(((v <= tc.slow_speed[1]) & ~static).float().mean()) < 0.27
    y0 = tr.y.clone()
    for _ in range(200):
        tr.step(0.005)
        assert float(tr.speed().max()) <= tc.speed[1] + 1e-6
    assert float((tr.y - y0)[static].abs().max()) == 0.0 and float((tr.y - y0)[~static].max()) < 0


def test_core_needs_neither_mujoco_nor_isaac():
    code = ("import sys\nfor m in ('mujoco','isaaclab','isaacsim'): sys.modules[m]=None\n"
            "from nero_vs_rl.isaac.core.task import GraspTask\nimport nero_vs_rl.isaac.sb3_vec_env, nero_vs_rl.isaac.policy_io\n"
            "from nero_vs_rl.algorithms.Networks import GraspPointGroupExtractor\nprint('ok')")
    assert "ok" in subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def test_isaac_config_consistent_with_task(cfg):
    ic = load_isaac_config()
    ratio = 1 / (cfg.control_freq * ic.physics_dt)
    assert abs(ratio - round(ratio)) < 1e-9 and len(ic.drive.stiffness) == len(ic.drive.damping) == 7
    assert all(0.08 <= d / k <= 0.12 for d, k in zip(ic.drive.damping, ic.drive.stiffness))       # 理论滞后在真机范围附近
