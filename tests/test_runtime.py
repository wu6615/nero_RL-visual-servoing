"""Closed-loop execution layer, offline: 100 Hz streamer, safety guard, grasp monitor, read-only NERO driver, MuJoCo-cell dry runs."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from nero_vs_rl import SIM_OUTPUT_DIR, TRAIN_CONFIG
from nero_vs_rl.sim2real.deploy.runtime.devices import RealClock, width_to_q
from nero_vs_rl.sim2real.deploy.runtime.episode import GraspMonitor
from nero_vs_rl.sim2real.deploy.runtime.safety import PROFILES
from nero_vs_rl.sim2real.deploy.runtime.streamer import JointStreamer
from nero_vs_rl.utils.config import load_config

MODEL = SIM_OUTPUT_DIR / "sac_dc1_v5" / "best" / "best_model.zip"


class FakeArm:
    def __init__(self, can_move=True):
        self.can_move, self.sent = can_move, []

    def send_joints(self, q):
        self.sent.append(np.array(q))


class ManualClock:
    def __init__(self):
        self.t, self.cb = 0.0, None

    def now(self):
        return self.t

    def every(self, period, fn):
        self.cb = fn


@pytest.fixture(scope="module")
def cfg():
    return load_config(TRAIN_CONFIG)[0]


# ---------------------------------------------------------------- streamer
def test_streamer_interpolates_then_holds():
    arm, clk = FakeArm(), ManualClock()
    st = JointStreamer(arm, 0.01, max_step_rad=np.full(7, 1.0))
    st.start(np.zeros(7), clk)
    st.set_target(np.full(7, 0.1), now=0.0, duration_s=0.1)
    for k in range(1, 16):
        clk.cb(0.01 * k)
    got = np.array(arm.sent)[:, 0]
    assert np.allclose(got[:10], 0.01 * np.arange(1, 11))         # 10 个插值点均匀走到目标
    assert np.allclose(got[10:], 0.1)                              # 到了之后重发终点 = 保持
    assert st.n_clamped == 0


def test_streamer_never_jumps_more_than_max_step():
    arm, clk = FakeArm(), ManualClock()
    st = JointStreamer(arm, 0.01, max_step_rad=np.full(7, 0.02))
    st.start(np.zeros(7), clk)
    st.set_target(np.full(7, 1.0), now=0.0, duration_s=0.01)      # 上游错误：一拍要跳 1 rad
    for k in range(1, 6):
        clk.cb(0.01 * k)
    steps = np.abs(np.diff(np.vstack([np.zeros(7), *arm.sent]), axis=0))
    assert steps.max() <= 0.02 + 1e-12 and st.n_clamped == 5


def test_streamer_rehearsal_sends_nothing():
    arm, clk = FakeArm(can_move=False), ManualClock()
    st = JointStreamer(arm, 0.01, max_step_rad=np.full(7, 1.0))
    st.start(np.zeros(7), clk)
    st.set_target(np.full(7, 0.05), 0.0, 0.05)
    for k in range(1, 6):
        clk.cb(0.01 * k)
    assert arm.sent == [] and np.allclose(st.q_cmd, 0.05)          # 只读预演：只算"本应发出"的指令


def test_real_clock_every_keeps_rate():
    clk, ticks = RealClock(), []
    clk.every(0.01, ticks.append)
    time.sleep(0.3)
    clk.stop()
    assert 20 <= len(ticks) <= 35 and np.median(np.diff(ticks)) == pytest.approx(0.01, abs=0.003)


# ---------------------------------------------------------------- grasp monitor
def test_grasp_monitor_needs_blocked_stable_jaw_and_lift():
    m = GraspMonitor(lift_height_m=0.06)
    # 闭合过程中开口还在变：不算抓住
    assert m.update(0.0, 0.08, 0.0) == (False, False) and m.update(0.0, 0.06, 0.0) == (False, False)
    # 被 4 cm 物体撑住、稳定 2 拍：抓住；抬高 6 cm：抬起
    m.update(0.0, 0.040, 0.0)
    m.update(0.0, 0.040, 0.0)
    assert m.update(0.0, 0.040, 0.01) == (True, False)
    assert m.update(0.0, 0.040, 0.075) == (True, True)
    # 夹空（开口收到接近 0）：不算
    m.reset()
    for _ in range(4):
        held, _ = m.update(0.0, 0.002, 0.0)
    assert not held


# ---------------------------------------------------------------- safety guard
@pytest.fixture(scope="module")
def guard_grasp(cfg):
    from nero_vs_rl.sim2real.deploy.common import RobotTwin
    from nero_vs_rl.sim2real.deploy.runtime.safety import SafetyGuard
    from nero_vs_rl.sim2real.deploy.runtime.session import stage_config
    twin = RobotTwin(stage_config(cfg, PROFILES["grasp"]))
    return SafetyGuard(twin, PROFILES["grasp"], belt_z_safe=cfg.belt_surface_z + 0.008)


def test_guard_accepts_home_and_rejects_bad_targets(guard_grasp):
    g = guard_grasp
    q0 = g.twin.env.q_home
    assert g.check_target(q0, q0, 0.05) == []
    assert any("单拍" in p for p in g.check_target(q0, q0 + 0.5, 0.05))      # 全速上限约 0.33～0.41 rad/拍
    assert g.check_target(q0, np.full(7, np.nan), 0.05) == ["关节目标不是 7 个有限数"]
    # TCP 压到带面以下：工作空间和碰撞都要报
    p, R = g.twin.controller.ik.fk(q0)
    q_low, _, _ = g.twin.controller.ik.solve(q0, p - [0, 0, p[2] - g.belt_z_safe + 0.02], R, iters=200)
    probs = g.check_pose(q_low, 0.05)
    assert any("工作空间" in x for x in probs) and any("碰到 belt_surface" in x for x in probs)


def test_guard_raises_target_until_fingers_clear_belt(guard_grasp):
    """Open gripper tilted 15 deg about the closing axis puts a fingertip ~24 mm below the TCP."""
    from nero_vs_rl.utils.geometry import expmap
    g = guard_grasp
    ik = g.twin.controller.ik
    p0, R0 = g.twin.home_tcp()
    R = R0 @ expmap(np.radians(15) * np.array([1.0, 0, 0]))
    q, _, _ = ik.solve(g.twin.env.q_home, [p0[0], p0[1], g.belt_z_safe + 0.010], R, iters=200)
    assert g.gripper_lowest_z(q, 0.05) < g.belt_z_safe                      # 修正前指尖在带面以下
    q2, fixes = g.project(q, 0.05)
    assert fixes and "抬高" in fixes[0]
    assert g.gripper_lowest_z(q2, 0.05) >= g.belt_z_safe + g.profile.finger_clearance_m - 5e-4
    assert g.check_pose(q2, 0.05) == []


def test_guard_raises_target_until_fingers_clear_rail(guard_grasp):
    """Open gripper lowered onto the far rail (the near one is out of reach): lifted over the rail top instead of stopped."""
    g = guard_grasp
    ik, m = g.twin.controller.ik, g.twin.model
    rail = m.geom("belt_rail_far").id
    p0, R0 = g.twin.home_tcp()
    x_rail = float(m.geom_pos[rail][0])
    q, _, _ = ik.solve(g.twin.env.q_home, [x_rail, p0[1], g.belt_z_safe + 0.012], R0, iters=200)
    assert any("belt_rail_far" in x for x in g.check_pose(q, 0.05))                   # 修正前：手指插进护栏
    q2, fixes = g.project(q, 0.05)
    assert any("护栏" in f and "抬高" in f for f in fixes)
    assert g.check_pose(q2, 0.05) == []
    P = g.twin.controller.gripper_points(q2, 0.05)
    over = np.abs(P[:, 0] - g.twin.cfg.belt_center_x) >= g.rail_inner
    assert over.any() and P[over, 2].min() >= g.rail_top_z + g.profile.finger_clearance_m - 5e-4


def test_rail_term_is_deploy_only(cfg):
    """Training env keeps v6's correction (no rail term): the controller's compute() never mentions the rail."""
    from nero_vs_rl.envs.dynamic_grasp import DynamicGraspEnv
    env = DynamicGraspEnv(cfg)
    try:
        env.reset(seed=1)
        for _ in range(15):
            env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
            assert not any("护栏" in f for f in env.controller.last_fixes)
    finally:
        env.close()


def test_guard_pulls_target_back_into_workspace(guard_grasp):
    """IK can land a few cm outside the workspace the controller clipped to; project() re-solves toward the box."""
    g = guard_grasp
    ik = g.twin.controller.ik
    p0, R0 = g.twin.home_tcp()
    q, _, _ = ik.solve(g.twin.env.q_home, [p0[0], p0[1], g.ws_hi[2] + 0.03], R0, iters=200)
    assert any("工作空间" in x for x in g.check_pose(q, 0.05))
    q2, fixes = g.project(q, 0.05)
    assert any("拉回工作空间" in f for f in fixes)
    assert ik.fk(q2)[0][2] <= g.ws_hi[2] + 0.002 and g.check_pose(q2, 0.05) == []


def test_guard_clips_to_soft_joint_limits(guard_grasp):
    g = guard_grasp
    q = g.twin.env.q_home.copy()
    q[3] = g.twin.controller.ik.hi[3]                                         # J4 顶到硬限位
    q2, fixes = g.project(q, 0.05)
    assert fixes and np.all(q2 >= g.q_lo - 1e-12) and np.all(q2 <= g.q_hi + 1e-12)   # 拉回/截断后一定在软限位内


def test_guard_uses_higher_belt_height(cfg, guard_grasp):
    assert guard_grasp.belt_z_safe == pytest.approx(cfg.belt_surface_z + 0.008)
    assert guard_grasp.ws_lo[2] == pytest.approx(cfg.belt_surface_z + 0.008 + PROFILES["grasp"].tcp_floor_above_belt_m)


# ---------------------------------------------------------------- NERO driver (fake SDK)
class _Msg(SimpleNamespace):
    pass


class FakeRobot:
    """Just enough of the pyAgxArm NERO driver; enable() must never be called."""
    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="agx"))

    def __init__(self, enabled=True):
        self.comm = SimpleNamespace(send=lambda *a, **k: self.frames.append(a))
        self.frames, self.enabled = [], enabled
        foc = SimpleNamespace(driver_enable_status=True, mode="width")
        self.gripper = SimpleNamespace(
            get_gripper_status=lambda: _Msg(msg=SimpleNamespace(value=0.1, mode="width", foc_status=foc), timestamp=time.time()),
            move_gripper_m=lambda value, force: self.comm.send(("grip", value, force)))

    def init_effector(self, kind):
        return self.gripper

    def create_comm(self):
        return self.comm

    def connect(self):
        pass

    def disconnect(self):
        pass

    def enable(self, *a):
        raise AssertionError("程序不允许自动使能")

    def get_joint_angles(self):
        return _Msg(msg=[0.0] * 7, timestamp=time.time())

    def get_arm_status(self):
        return _Msg(msg=SimpleNamespace(arm_status=0, err_code=0, ctrl_mode=1, teach_status=0), timestamp=time.time())

    def get_driver_states(self, j):
        return _Msg(msg=SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=self.enabled)))

    def has_comm_error(self):
        return False

    def move_js(self, q):
        self.comm.send(("js", q))

    def electronic_emergency_stop(self):
        self.comm.send(("estop",))

    def set_speed_percent(self, p):
        self.comm.send(("speed", p))


def _controller(monkeypatch, allow_motion, enabled=True):
    from nero_vs_rl.sim2real.hardware import nero_arm, nero_control
    robot = FakeRobot(enabled)
    monkeypatch.setattr(nero_arm, "check_can_interface", lambda ch: dict(channel=ch))
    monkeypatch.setattr(nero_control.NeroController, "_create_robot", lambda self: robot)
    arm = nero_control.NeroController(allow_motion=allow_motion).connect()
    return arm, robot


def test_nero_rehearsal_cannot_send_anything(monkeypatch):
    arm, robot = _controller(monkeypatch, allow_motion=False, enabled=False)
    try:
        with pytest.raises(RuntimeError, match="只读预演"):
            arm.send_joints(np.zeros(7))
        with pytest.raises(RuntimeError, match="只读预演"):
            arm.send_gripper(0.05)
        arm.emergency_stop()                                   # 预演里急停是空操作
        assert robot.frames == []
        assert arm.faults() == []                              # 预演不要求使能（操作者可能在拖动）
    finally:
        arm.close()


def test_nero_execute_requires_enabled_joints_and_streams_js(monkeypatch):
    arm, robot = _controller(monkeypatch, allow_motion=True, enabled=False)
    try:
        assert any("未使能" in f for f in arm.faults())
        with pytest.raises(RuntimeError, match="没准备好"):
            arm.prepare_motion()                               # 没使能就不准备运动（也不会去使能）
        robot.enabled = True
        arm.prepare_motion()
        arm.send_joints(np.full(7, 0.1))
        arm.send_gripper(0.2)                                  # 超出行程的开口被截到 0.10
        assert robot.frames[0][0] == ("speed", 100) and robot.frames[1][0][0] == "js"
        assert robot.frames[2][0][1] == pytest.approx(0.10)
    finally:
        arm.close()


# ---------------------------------------------------------------- MuJoCo-cell dry runs
def _run_cell(cfg, profile, seeds, policy, wait_s=30.0, park_object=False):
    from nero_vs_rl.sim2real.deploy.runtime.session import LoopSession
    from nero_vs_rl.sim2real.hardware.sim_workcell import SimWorkcell
    cell = SimWorkcell(cfg)
    cell.reset(seed=seeds[0])
    if park_object:
        cell.park_object()
    s = LoopSession(cfg, PROFILES[profile], cell.arm, cell.camera, cell.clock, belt_z_safe=cfg.belt_surface_z, latency_s=0.0)
    out = []
    try:
        for i, seed in enumerate(seeds):
            if i:
                cell.reset(seed=seed)
                s.streamer.reset_to(cell.arm.joints(), cell.clock.now())
            truth = []
            res = s.episode(policy, wait_s=wait_s, on_tick=lambda ep, row: truth.append(cell.truth()["lifted"]) and False).run()
            out.append((res, any(truth), s))
        return out
    finally:
        s.close()
        cell.close()


def test_sim_camera_latency_shows_the_past_scene(cfg):
    """With latency L the frame at time t is the ideal frame of time t - L (object moving on the belt), stamped t."""
    from nero_vs_rl.sim2real.hardware.sim_workcell import CameraEffects, SimWorkcell
    ideal, late = SimWorkcell(cfg), SimWorkcell(cfg, camera_effects=CameraEffects(latency_s=0.1))
    try:
        for c in (ideal, late):
            c.reset(seed=4)
        ideal.clock.sleep_until(0.4)
        past = ideal.camera.latest().depth
        ideal.clock.sleep_until(0.5)
        now = ideal.camera.latest().depth
        late.clock.sleep_until(0.5)
        f = late.camera.latest()
        assert f.stamp == pytest.approx(0.5)
        diff = lambda a, b: float((np.abs(a - b) > 0.01).mean())           # 深度差 > 1 cm 的像素比例
        assert diff(f.depth, past) < 0.002 and diff(now, past) > 0.01       # 物体在动，两帧不同；延迟帧 = 过去那帧
    finally:
        ideal.close(); late.close()


def test_sim_camera_depth_noise_scales_with_distance(cfg):
    from nero_vs_rl.sim2real.hardware.sim_workcell import CameraEffects, SimWorkcell
    c = SimWorkcell(cfg, camera_effects=CameraEffects(depth_noise_m=0.003, edge_dropout=1.0))
    try:
        c.reset(seed=4)
        flat = np.full((40, 40), 0.6, np.float32)
        flat[:, 20:] = 0.3                                                    # 中间一条 30 cm 的深度跳变
        out = c.camera.degrade(flat)
        assert out[:, 18:22][out[:, 18:22] == 0].size > 0                      # 跳变边缘被置为无效
        assert np.std(out[:, 25:][out[:, 25:] > 0] - 0.3) == pytest.approx(0.003, rel=0.2)
        assert np.std(out[:, :15] - 0.6) == pytest.approx(0.012, rel=0.2)     # (0.6/0.3)² × 3 mm
    finally:
        c.close()


def test_place_static_object_and_set_arm(cfg):
    from nero_vs_rl.sim2real.hardware.sim_workcell import SimWorkcell
    c = SimWorkcell(cfg)
    try:
        c.reset(seed=2)
        q = c.arm.joints() + 0.05
        c.place_static_object(-0.40, 0.12, 0.3)
        c.set_arm(q)
        c.clock.sleep_until(1.0)
        p = c.data.xpos[c.env.obj["body"]]
        assert np.allclose(p[:2], [-0.40, 0.12], atol=2e-3) and c.belt_speed() == 0.0
        assert np.abs(c.arm.joints() - q).max() < 0.02
    finally:
        c.close()


def test_episode_waits_without_target_and_never_moves(cfg):
    calls = []
    res, _, s = _run_cell(cfg, "grasp", [0], lambda obs: calls.append(1) or np.zeros(7), wait_s=1.0, park_object=True)[0]
    assert res.outcome == "no_target" and calls == []           # 没看到物体：策略一次都没被调用
    assert all(r["phase"] == "WAIT" for r in res.rows)


@pytest.mark.skipif(not MODEL.exists(), reason="没有 v5 模型")
def test_dry_run_grasp_profile_keeps_fingers_above_belt_and_lifts(cfg):
    from nero_vs_rl.sim2real.deploy.runtime.session import load_policy
    runs = _run_cell(cfg, "grasp", [20000, 20001, 20003], load_policy(str(MODEL)))
    g = runs[0][2].guard
    for res, truth, _ in runs:
        assert res.outcome in ("lifted", "timeout", "stopped")
        for r in res.rows:
            if "q_tgt" in r and not r.get("problems"):
                g_open = max(width_to_q(r["width"]), width_to_q(r["grip_cmd"]))      # 同 episode._servo 的判据
                assert g.gripper_lowest_z(r["q_tgt"], g_open) >= g.belt_z_safe + g.profile.finger_clearance_m - 1e-3
        assert (res.outcome == "lifted") == truth or res.outcome == "stopped"   # 抓起判断和仿真真值一致
    # v5 是旧规则下训的；v8 规则（对正、两指合到物体宽度才算夹住）下这 3 个种子只抓起 1 个，参数换回旧值也一样（2026-11-26 对照）
    assert sum(res.outcome == "lifted" for res, _, _ in runs) >= 1


@pytest.mark.skipif(not MODEL.exists(), reason="没有 v5 模型")
def test_dry_run_follow_profile_never_closes_or_descends(cfg):
    from nero_vs_rl.sim2real.deploy.runtime.session import load_policy
    res, truth, s = _run_cell(cfg, "follow", [20000], load_policy(str(MODEL)))[0]
    assert not truth and res.outcome != "lifted"
    servo = [r for r in res.rows if "q_tgt" in r]
    assert servo and all(r["grip_cmd"] == pytest.approx(0.10) for r in servo)
    floor = cfg.belt_surface_z + PROFILES["follow"].tcp_floor_above_belt_m
    for r in servo:
        p, _ = s.twin.controller.ik.fk(r["q_tgt"])
        assert p[2] >= floor - 0.006


def test_step5_motion_check_in_sim(tmp_path):
    from nero_vs_rl.sim2real.deploy import step5_motion_check
    rep = step5_motion_check.main(["--sim", "--no-viewer", "--gripper", "--out-root", str(tmp_path)])   # 不往数据盘写
    assert len(list(tmp_path.glob("*_motion_check/report.json"))) == 1
    assert rep["problems"] == [] and rep["tracking"]["stream_hz"] == pytest.approx(100, rel=0.05)
    assert rep["gripper"]["cmd_0.05"] == pytest.approx(0.05, abs=0.003)


def test_preflight_waits_for_joint_history_before_judging_stationary():
    # 真机 step5 不开相机：刚连上 CAN 时关节历史不足 1 s，静止判定前几次必然是 False
    from nero_vs_rl.sim2real.deploy.lab import preflight

    class WarmingArm:
        def __init__(self, ready_after):
            self.calls, self.ready_after = 0, ready_after

        def faults(self):
            return []

        def stationary(self):
            self.calls += 1
            return self.calls > self.ready_after

    dev = SimpleNamespace(arm=WarmingArm(ready_after=5), camera=None)
    assert preflight(dev, cfg=None, need_start_pose=False) == []
    dev = SimpleNamespace(arm=WarmingArm(ready_after=10**9), camera=None)    # 真在动：3 s 后仍判不通过
    t0 = time.monotonic()
    assert preflight(dev, cfg=None, need_start_pose=False) == ["机械臂静止"]
    assert 2.9 < time.monotonic() - t0 < 4.0


def test_nero_connect_waits_for_status_and_driver_feedback(monkeypatch):
    # 真机：刚连上时只有关节角，状态 / 驱动反馈晚几毫秒才到；connect() 返回前要等齐，开机检查才不会误报
    from nero_vs_rl.sim2real.hardware import nero_arm, nero_control
    robot, t_ready = FakeRobot(True), time.monotonic() + 0.1
    real_status, real_driver = robot.get_arm_status, robot.get_driver_states
    robot.get_arm_status = lambda: real_status() if time.monotonic() > t_ready else None
    robot.get_driver_states = lambda j: real_driver(j) if time.monotonic() > t_ready else None
    monkeypatch.setattr(nero_arm, "check_can_interface", lambda ch: dict(channel=ch))
    monkeypatch.setattr(nero_control.NeroController, "_create_robot", lambda self: robot)
    arm = nero_control.NeroController(allow_motion=True).connect()
    try:
        assert arm.faults() == []
    finally:
        arm.close()


def test_ask_keeps_preview_alive_while_waiting_for_enter(monkeypatch):
    # 回合间等终端回车时，预览窗口要一直刷新，否则桌面会判"无响应"（11-24 现场被强制退出）
    import os
    import sys
    import threading
    from nero_vs_rl.sim2real.deploy.lab import ask

    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r))
    preview = SimpleNamespace(n=0)
    preview.idle = lambda: (setattr(preview, "n", preview.n + 1), time.sleep(0.01))
    threading.Timer(0.3, lambda: os.write(w, "q\n".encode())).start()
    assert ask("提示：", preview) == "q"
    assert preview.n >= 10
    os.close(w)
    with pytest.raises(EOFError):                             # 终端关了（EOF）和 input() 一样报 EOFError
        ask("提示：", preview)
