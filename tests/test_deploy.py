"""Deployment offline stage: shared grasp memory, joint interpolation, and sim record -> replay against ground truth."""
import numpy as np
import pytest

from nero_vs_rl import TRAIN_CONFIG
from nero_vs_rl.envs.observation import GraspTracker
from nero_vs_rl.sim2real.hardware.nero_arm import interp_joints
from nero_vs_rl.utils.config import GraspObsConfig, load_config


def test_tracker_extrapolates_with_fitted_velocity():
    cfg = GraspObsConfig(memory_mode="extrapolate", memory_max_age_steps=5, vel_window=8)
    tr = GraspTracker(cfg, control_dt=0.1)
    v = np.array([0.0, -0.05, 0.0])
    G = np.tile(np.eye(4), (2, 1, 1))
    for k in range(6):                                       # 匀速运动的物体，连看 6 帧
        p = np.array([-0.38, 0.2, -0.1]) + v * 0.1 * k
        Gk = G.copy()
        Gk[:, :3, 3] = p
        out, sc, vis = tr.update(0.1 * k, Gk, np.ones(2), p)
        assert vis and tr.mem_age == 0
    assert np.allclose(tr.est_obj_vel, v, atol=1e-9)
    out, _, vis = tr.update(0.6, None)                       # 看不见 3 帧：沿估计速度外推
    out, _, vis = tr.update(0.7, None)
    out, _, vis = tr.update(0.8, None)
    assert not vis and tr.mem_age == 3
    assert np.allclose(out[0, :3, 3], p + v * 0.1 * 3)
    for _ in range(3):
        out, _, _ = tr.update(1.0, None)
    assert len(out) == 0                                     # 超过记忆时长后清空


def test_interp_joints_by_timestamp():
    hist = [(0.0, np.zeros(7)), (0.01, np.ones(7)), (0.02, 3 * np.ones(7))]
    assert np.allclose(interp_joints(hist, 0.005), 0.5)
    assert np.allclose(interp_joints(hist, 0.015), 2.0)
    assert np.allclose(interp_joints(hist, -1.0), 0.0) and np.allclose(interp_joints(hist, 9.0), 3.0)


@pytest.fixture(scope="module")
def sim_record(tmp_path_factory):
    from nero_vs_rl.sim2real.deploy import step1_record
    out = tmp_path_factory.mktemp("deploy") / "rec"
    # 种子 4：长方体在带中间、带速 14～28 cm/s（v7 起手指长度随机化多抽一个随机数，种子 0 变成了贴近护栏、很慢的圆柱，
    #   位置误差 4.3 mm；种子 1～5 都是 1.0～1.1 mm）
    step1_record.main(["--sim", "--seconds", "12", "--seed", "4", "--out", str(out), "--no-preview"])
    return out


def test_record_format(sim_record):
    from nero_vs_rl.sim2real.deploy.common import load_record, read_depth
    meta, frames = load_record(sim_record)
    assert meta["source"] == "sim" and len(frames) == 120
    d = read_depth(sim_record, frames[0])
    assert d.shape == (400, 640) and d.max() < 2.0 and (d > 0).mean() > 0.3      # mm → m 往返正确
    assert "truth" in frames[0] and len(frames[0]["joints"]) == 7


def test_replay_perception_matches_ground_truth(sim_record, tmp_path):
    """Sim depth is noise-free, so the detector must recover the true object closely; obs must fit the training space."""
    import torch
    from stable_baselines3 import SAC

    from nero_vs_rl.algorithms.Networks import GraspPointGroupExtractor
    from nero_vs_rl.envs import DynamicGraspEnv
    from nero_vs_rl.sim2real.deploy.step2_replay import replay

    cfg, _ = load_config(TRAIN_CONFIG)
    env = DynamicGraspEnv(cfg)
    model = SAC("MultiInputPolicy", env, buffer_size=10, learning_starts=0, device="cpu", seed=0,
                policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=[32, 32]))
    model.save(tmp_path / "tiny.zip")
    env.close()
    torch.set_num_threads(2)
    s = replay(sim_record, str(tmp_path / "tiny.zip"), str(TRAIN_CONFIG), video=False)
    assert s["camera_latency_s"] == 0.0                     # 仿真录制不套用真机测得的相机延迟
    assert s["obs_all_in_space"]
    assert s["visible_frac"] > 0.2
    assert s["truth_center_err_mm_median"] < 3.0
    assert s["truth_size_err_mm_median"] < 2.0
    assert s["truth_type_acc"] >= 0.9
    assert s["policy_toward_frames"] > 0                    # 策略查询链路走通（未训练模型，不看朝向比例）
    assert len(s["objects"]) >= 1
    lo, hi = cfg.trajectory.speed
    timed = [o for o in s["objects"] if "speed_cm_s" in o]   # 高速物体可见帧太少时没有速度估计
    assert timed
    for o in timed:                                          # v7 起有静止 / 慢带物体：方向只对动起来的物体查
        assert o["speed_cm_s"] < 1.2 * hi * 100
        if o["speed_cm_s"] >= 2.0:
            assert o["direction_dev_deg"] < 10                # 沿 −Y
    assert (sim_record / "replay_report.json").exists()


def test_detector_splits_objects_and_ignores_empty_belt():
    """Empty belt -> no detection; object gone for LOST_FRAMES frames and back -> counted as a new object."""
    import mujoco

    from nero_vs_rl.sim2real.deploy.common import RobotTwin
    from nero_vs_rl.sim2real.deploy.perception import LOST_FRAMES, BeltObjectDetector
    from nero_vs_rl.utils.geometry import make_T

    cfg, _ = load_config(TRAIN_CONFIG)
    tw = RobotTwin(cfg)
    env = tw.env
    fy = cfg.camera.intrinsics[1]
    K = np.array([[fy, 0, cfg.camera.width / 2], [0, fy, cfg.camera.height / 2], [0, 0, 1.0]])
    det = BeltObjectDetector(cfg, K, np.zeros(5), tw.max_width)
    env.reset(seed=2)                                        # 这个种子开局物体就完整在画面里
    tw.set_joints(env.agent.arm_qpos(env.data))
    p, R = tw.tcp_pose()
    T_bc, T_bt = tw.T_base_cam_cv(), make_T(R, p)
    qa = env.obj["qadr"]
    home = env.data.qpos[qa:qa + 3].copy()

    def look(pos):
        env.data.qpos[qa:qa + 3] = pos
        mujoco.mj_forward(env.model, env.data)
        return det.detect(env.render_wrist_rgbd()[1], T_bc, T_bt)

    d = look(home)
    assert d is not None and not d.truncated and det.new_object
    first = det.object_id
    assert look(home) is not None and not det.new_object and det.object_id == first
    for _ in range(LOST_FRAMES):
        assert look([2.0, 2.0, -0.5]) is None               # 只剩带面：不能把带面/护栏当物体
    assert look(home) is not None and det.new_object and det.object_id == first + 1
    tw.close()


def test_real_recorder_rate_with_fake_devices(tmp_path, monkeypatch):
    """The real-device path (no hardware): 24 fps camera stamps -> ~10 Hz recording, joints taken at frame time."""
    import time

    from nero_vs_rl.sim2real.deploy import step1_record
    from nero_vs_rl.sim2real.deploy.common import load_record
    from nero_vs_rl.sim2real.hardware import dc1_camera, nero_arm
    from nero_vs_rl.sim2real.hardware.dc1_camera import RGBDFrame

    class FakeCam:
        def __init__(self, cfg):
            self.t = 100.0

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass

        def camera_info(self):
            return dict(depth_intrinsics=dict(width=640, height=400, fx=475.0, fy=475.0, cx=320.0, cy=200.0, dist=[0.0] * 5))

        def read(self):
            time.sleep(0.002)
            self.t += 1 / 24
            return RGBDFrame(np.zeros((720, 1280, 3), np.uint8), np.full((400, 640), 0.4, np.float32), self.t)

    class FakeArm:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass

        def joints_at(self, t):
            return np.full(7, t)

        def history(self, since=-np.inf):
            self.k = getattr(self, "k", 0) + 1
            return [(float(self.k), np.full(7, float(self.k)))]

    monkeypatch.setattr(dc1_camera, "DC1Camera", FakeCam)
    monkeypatch.setattr(nero_arm, "NeroArmReader", FakeArm)
    out = tmp_path / "real_rec"
    step1_record.main(["--seconds", "0.6", "--out", str(out), "--no-preview"])
    meta, frames = load_record(out)
    st = np.array([f["stamp"] for f in frames])
    assert meta["source"] == "real" and len(frames) > 10
    assert abs(np.median(np.diff(st)) - 0.1) < 0.03                       # 约 10 Hz
    assert all(np.allclose(f["joints"], f["stamp"]) for f in frames)      # 关节按帧时间戳取
    hist = np.load(out / "joints_hist.npz")                                # 100 Hz 关节历史也存下来了
    assert len(hist["t"]) == len(frames) and hist["q"].shape == (len(frames), 7)


def test_check_pose_reports_offset_from_sim_home(monkeypatch):
    """--check-pose (read-only): sim home joints -> zero offset and OK; a shifted pose -> not OK."""
    from nero_vs_rl.sim2real.deploy import step1_record
    from nero_vs_rl.sim2real.deploy.common import RobotTwin
    from nero_vs_rl.sim2real.hardware import nero_arm

    cfg, _ = load_config(TRAIN_CONFIG)
    tw = RobotTwin(cfg)
    q_home = tw.env.q_home.copy()
    err = step1_record.pose_error(tw, q_home)
    assert np.abs(err["dp_mm"]).max() < 3 and err["tilt_deg"] < 0.5 and abs(err["yaw_deg"]) < 0.5
    assert step1_record.pose_ok(err, cfg)
    q_bad = q_home.copy()
    q_bad[0] += 0.4                                          # 转基座关节：TCP 横移十几厘米
    assert not step1_record.pose_ok(step1_record.pose_error(tw, q_bad), cfg)
    tw.close()

    class FakeArm:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass

        def joints(self):
            return q_home

    monkeypatch.setattr(nero_arm, "NeroArmReader", FakeArm)
    out = step1_record.main(["--check-pose", "--seconds", "0.3"])
    assert step1_record.pose_ok(out, cfg)


def test_latency_estimate_recovers_injected_delay(tmp_path):
    """Sim recording with the arm moving; frames relabelled 80 ms late -> step3 should find ~80 ms."""
    import json

    from nero_vs_rl.sim2real.deploy import step1_record
    from nero_vs_rl.sim2real.deploy.step3_measure_latency import estimate

    d = tmp_path / "lat"
    step1_record.main(["--sim", "--sim-motion", "scripted", "--seconds", "30", "--seed", "4", "--out", str(d),
                       "--no-preview"])
    fr = [json.loads(line) for line in (d / "frames.jsonl").read_text().splitlines()]
    t, off = [], 0.0
    for f in fr:                                             # 各回合时间从 0 重来：拼成单调时间轴
        s = f["stamp"] + off
        if t and s <= t[-1]:
            off += t[-1] - f["stamp"] + 1.0
            s = f["stamp"] + off
        t.append(s)
    t, q = np.array(t), np.array([f["joints"] for f in fr])
    np.savez(d / "joints_hist.npz", t=t, q=q)                 # 关节历史 = 真实时刻
    L = 0.08
    with open(d / "frames.jsonl", "w") as fo:                 # 帧到达时刻 = 拍摄时刻 + 80 ms
        for f, tt in zip(fr, t):
            f["stamp"] = float(tt + L)
            f["joints"] = [float(np.interp(tt + L, t, q[:, j])) for j in range(7)]
            fo.write(json.dumps(f) + "\n")
    r = estimate(d, str(TRAIN_CONFIG))
    assert r["joints_source"] == "100Hz" and r["n_tracks"] >= 1
    assert abs(r["latency_s"] - L) <= 0.03
    assert r["rms_mm_best"] < r["rms_mm_at_zero"]
