"""NERO demonstrations, offline: joint commands -> RL action, NERO run-log parsing, recording + log -> demos.npz."""
import json

import numpy as np
import pytest

from nero_vs_rl.envs.dynamic_grasp import DynamicGraspEnv
from nero_vs_rl.sim2real.deploy.runtime.devices import q_to_width
from nero_vs_rl.sim2real.demos.actions import joints_to_action
from nero_vs_rl.sim2real.demos.nero_log import CommandTimeline, load_run
from nero_vs_rl.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    from nero_vs_rl import TRAIN_CONFIG
    return load_config(TRAIN_CONFIG)[0]                           # 现场配置：动作缩放和训练一致


def test_joints_to_action_inverts_the_rl_controller(cfg):
    """Command one control step with a known action, convert the joint targets back: the same action comes out."""
    env = DynamicGraspEnv(cfg)
    ctl, rng = env.controller, np.random.default_rng(3)
    try:
        for seed in range(6):
            env.reset(seed=seed)
            ctl.reset(env.data)
            q0 = ctl.q_cmd.copy()
            a = np.r_[rng.uniform(-0.5, 0.5, 6), rng.uniform(-1, 1)]
            q1, grip_q = ctl.compute(env.data, a)
            if ctl.last_fixes:                                    # 修正过的目标不再是这个动作，跳过
                continue
            got = joints_to_action(ctl.ik.fk, q0, q1, q_to_width(grip_q), env.cfg)
            assert np.allclose(got.action[:6], a[:6], atol=0.03), (got.action, a)
            assert got.action[6] == pytest.approx(a[6], abs=1e-6)
            assert not got.clipped
    finally:
        env.close()


def test_joints_to_action_flags_steps_the_policy_cannot_make(cfg):
    env = DynamicGraspEnv(cfg)
    try:
        env.reset(seed=0)
        q0 = env.agent.arm_qpos(env.data)
        q1 = q0.copy()
        q1[0] += 0.3                                              # J1 一步转 0.3 rad：远超每步 5 cm / 0.1 rad
        got = joints_to_action(env.controller.ik.fk, q0, q1, 0.1, env.cfg)
        assert got.clipped and np.all(np.abs(got.action) <= 1.0) and np.abs(got.raw[:6]).max() > 1.0
    finally:
        env.close()


def _fake_nero_cycle(d, t, q, stages, gripper, confirmed=True, track=(0, 113)):
    d.mkdir(parents=True)
    np.savez(d / "execution_trace.npz", dispatch_start_monotonic_s=t, dispatched_q_rad=q, sample_stage=np.array(stages))
    (d / "execution_trace_summary.json").write_text(json.dumps(dict(gripper_commands=[
        dict(sdk_start_monotonic=tg, width_m=w, succeeded=True) for tg, w in gripper],
        target_identity=dict(track_generation=track[0], track_id=track[1]))))
    (d / "events.json").write_text(json.dumps([dict(event="grip_confirmed_from_width_force")] if confirmed else []))
    (d / "report.json").write_text(json.dumps(dict(status="cycle_completed_from_feedback")))


def test_nero_log_stages_close_time_and_timeline(tmp_path):
    t = np.arange(0, 3.0, 0.01) + 100.0
    q = np.outer(np.arange(len(t)), np.ones(7)) * 1e-3
    stages = ["smooth_intercept_follow_brake"] * 200 + ["stationary_hold"] * 100
    _fake_nero_cycle(tmp_path / "run" / "cycle_001", t, q, stages, [(101.0, 0.0), (102.5, 0.1)])
    (c,) = load_run(tmp_path / "run")
    assert c.intercept.name == "smooth_intercept_follow_brake" and c.intercept.t_end == pytest.approx(101.99)
    assert c.t_close == pytest.approx(101.0) and c.grip_confirmed and c.target == (0, 113)
    tl = CommandTimeline([c])
    assert tl.q_at(100.505)[0] == pytest.approx(0.0505, abs=1e-6)       # 100 Hz 指令之间线性插值
    assert tl.q_at(99.0) is None and tl.q_at(200.0) is None             # 指令时间线之外：调用方用实测关节
    assert tl.width_at(100.5) == 0.1 and tl.width_at(101.2) == 0.0 and tl.width_at(103.0) == 0.1


def test_build_demos_from_a_sim_recording_and_a_fake_nero_log(tmp_path):
    """End to end on the MY_RL_test layout: recording in the run's data dir, NERO cycle logs copied into nero_cycles/."""
    from nero_vs_rl import TRAIN_CONFIG
    from nero_vs_rl.sim2real.deploy import step1_record
    from nero_vs_rl.sim2real.demos import build_demos

    data = tmp_path / "nero_demos" / "continuous_20261126_000000_1"   # 录制格式和 NERO 腕部录制相同
    step1_record.main(["--sim", "--sim-motion", "scripted", "--seconds", "6", "--out", str(data), "--no-preview"])
    frames = [json.loads(line) for line in (data / "frames.jsonl").read_text().splitlines()]
    ts = np.array([f["stamp"] for f in frames])
    qs = np.array([f["joints"] for f in frames])
    t = np.arange(0.5, 5.5, 0.01)
    q_cmd = np.array([[np.interp(x + 0.1, ts, qs[:, j]) for j in range(7)] for x in t])   # 指令领先实测 0.1 s
    _fake_nero_cycle(data / "nero_cycles" / "cycle_001", t, q_cmd, ["smooth_intercept_follow_brake"] * len(t), [(3.0, 0.0)])
    (data / "belt_speed.json").write_text(json.dumps(dict(per_target=[dict(          # NERO 侧从运行日志抽的带速
        track_generation=0, track_id=113, speed_m_s=0.096, velocity_base_m_s=[-0.005, -0.096, 0.0])], timeline=[])))

    s = build_demos.main(["--run", str(data), "--latency", "0", "--pre-s", "0.5"])
    assert s["episodes"] == 1 and s["success"] == 1 and s["ticks"] >= 20
    assert s["command_lag_s"] == pytest.approx(0.1, abs=0.03)
    z = np.load(data / "demos.npz")                                # 写在同一个数据目录里
    space = DynamicGraspEnv(load_config(TRAIN_CONFIG)[0]).observation_space
    for key, sub in space.spaces.items():
        assert z[f"obs_{key}"].shape[1:] == sub.shape
    assert np.all(np.abs(z["action"]) <= 1.0) and z["episode_last"][-1]
    assert np.allclose(z["obs_action"][1:], z["action"][:-1])               # 观测里的上一步动作 = 上一拍示范动作
    assert z["episode_belt_speed"][0] == pytest.approx(0.096) and z["episode_belt_velocity"].shape == (1, 3)
    assert z["obj_vel_est"].shape == (len(z["action"]), 3) and s["per_episode"][0]["belt_speed_cm_s"] == pytest.approx(9.6)


def test_resolve_inputs_old_layout_writes_outside_the_nero_run(tmp_path):
    from nero_vs_rl import REAL_DATA_DIR
    from nero_vs_rl.sim2real.demos.build_demos import resolve_inputs
    run = tmp_path / "continuous_20261126_000000_1"
    (run / "wrist_dc1").mkdir(parents=True)
    (run / "wrist_dc1" / "frames.jsonl").write_text("")
    record, cycles, out = resolve_inputs(run)                      # 旧结构：录制在 NERO 运行目录的 wrist_dc1/
    assert record == run / "wrist_dc1" and cycles == run and out == REAL_DATA_DIR / "nero_demos" / run.name
    with pytest.raises(SystemExit, match="录腕部 DC1"):
        resolve_inputs(tmp_path / "empty")
