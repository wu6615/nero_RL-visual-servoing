"""Calibration pipeline, offline: frame conventions, hand-eye math, sim dry run, belt and blind-zone estimators."""
import numpy as np
import pytest

from nero_vs_rl.envs import DynamicGraspEnv
from nero_vs_rl.sensors import camera as cam
from nero_vs_rl.sim2real.calibration import handeye as he
from nero_vs_rl.sim2real.calibration.board import Chessboard, canonical_order, detect
from nero_vs_rl.sim2real.calibration.common import RobotFK, read_json
from nero_vs_rl.sim2real.calibration.sim_bridge import camera_overrides, cv_cam_to_mujoco, mujoco_cam_to_cv
from nero_vs_rl.sim2real.calibration.step4_measure_blind_zone import estimate_min_depth
from nero_vs_rl.sim2real.calibration.step5_measure_belt import belt_edges, depth_to_base, fit_plane
from nero_vs_rl.sim2real.hardware.nero_arm import euler_xyz_to_R
from nero_vs_rl.sim2real.hardware.sim_devices import DEFAULT_TRUE_MOUNT, SimDC1Camera, SimNeroArm, SimRig, mount_T
from nero_vs_rl.utils.config import EnvConfig, load_config

import cv2
from pathlib import Path


@pytest.fixture(scope="module")
def q_home():
    return DynamicGraspEnv().q_home


def test_cv_mujoco_camera_roundtrip():
    X = mount_T(**DEFAULT_TRUE_MOUNT)
    pos, quat = cv_cam_to_mujoco(X)
    assert np.allclose(mujoco_cam_to_cv(pos, quat), X, atol=1e-9)


def test_calibrated_camera_projects_like_opencv_inside_env(q_home):
    """The env, configured from a calibration result, must project exactly like OpenCV with the same K/X."""
    X = mount_T(**DEFAULT_TRUE_MOUNT)
    intr = dict(width=640, height=400, fx=420.0, fy=421.0, cx=318.0, cy=203.0)
    env = DynamicGraspEnv(EnvConfig.from_dict({"camera": camera_overrides(X, intr)}))
    env.agent.set_state(env.data, q_home, 0.05)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    T_base_cam = RobotFK().body_T(q_home) @ X
    in_cam = np.array([[0.02, -0.01, 0.30], [-0.05, 0.03, 0.45], [0, 0, 0.2]])      # OpenCV 相机系下的点
    pts = in_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    uv_sim, depth = cam.project(pts, env.data.cam_xpos[env.agent.camera], env.data.cam_xmat[env.agent.camera].reshape(3, 3),
                                env.cfg.camera)
    T_cam_base = he.inv(T_base_cam)
    uv_cv, _ = cv2.projectPoints(pts, cv2.Rodrigues(T_cam_base[:3, :3])[0], T_cam_base[:3, 3],
                                 np.array([[420.0, 0, 318], [0, 421.0, 203], [0, 0, 1]]), np.zeros(5))
    assert np.allclose(uv_sim, uv_cv.reshape(-1, 2), atol=1e-6)
    assert np.allclose(depth, [0.30, 0.45, 0.2], atol=1e-9)


def test_euler_matches_sdk_convention():
    # pyAgxArm 'sxyz'：R = Rz(yaw) Ry(pitch) Rx(roll)
    R = euler_xyz_to_R(0.3, -0.2, 1.1)
    assert np.isclose(np.arctan2(R[2, 1], R[2, 2]), 0.3) and np.isclose(np.arctan2(R[1, 0], R[0, 0]), 1.1)


def test_handeye_recovers_mount_from_noisy_synthetic_data(q_home):
    rng = np.random.default_rng(0)
    board, fk = Chessboard(), RobotFK()
    X = mount_T(**DEFAULT_TRUE_MOUNT)
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    B = np.eye(4)
    B[:3, :3] = np.diag([1.0, -1, -1])
    B[:3, 3] = [-0.50, 0.08, -0.116]
    rig = SimRig(board, T_base_board=B, X_true=X, seed=1)
    samples = []
    for i, q in enumerate(rig.random_view_joints(q_home, 15)):
        T_bg = fk.body_T(q)
        T_cb = he.inv(X) @ he.inv(T_bg) @ B
        uv, _ = cv2.projectPoints(board.object_points(), cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, np.zeros(5))
        corners = uv.reshape(-1, 2) + rng.normal(0, 0.3, (board.cols * board.rows, 2))
        from nero_vs_rl.sim2real.calibration.board import solve_pnp
        T_pnp, rms = solve_pnp(corners, board, K, np.zeros(5))
        samples.append(he.Sample(fk.body_T(q + rng.normal(0, 2e-5, 7)), T_pnp, corners, rms, i))
    rep = he.calibrate(samples, board, K, np.zeros(5))
    assert np.linalg.norm(rep["X"][:3, 3] - X[:3, 3]) < 1e-3
    assert he.rot_deg(rep["X"][:3, :3].T @ X[:3, :3]) < 0.2
    assert rep["consistency"]["trans_max_m"] < 0.003


def test_corner_order_is_canonical_after_reversal(q_home):
    rig = SimRig()
    arm, camera = SimNeroArm(rig), SimDC1Camera(rig)
    arm.move_to(rig.random_view_joints(q_home, 1)[0])
    img = camera.read().color
    c = detect(img, rig.board)
    assert c is not None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    assert np.allclose(canonical_order(c[::-1].copy(), gray, rig.board), c)


def test_full_dry_run_capture_solve_apply_verify(tmp_path):
    from nero_vs_rl.sim2real.calibration import step1_capture_handeye, step2_solve_handeye, step3_verify_in_sim
    session = tmp_path / "handeye_sim"
    step1_capture_handeye.main(["--sim", "--auto", "14", "--out", str(session)])
    site = tmp_path / "site.yaml"
    res = step2_solve_handeye.main(["--session", str(session), "--apply", "--site-config", str(site)])
    truth = read_json(session / "camera_info.json")["sim_truth"]
    assert res["accepted"]
    assert np.linalg.norm(np.asarray(res["T_gripper_base_color_cv"])[:3, 3] - np.asarray(truth["X_color"])[:3, 3]) < 1e-3
    assert np.linalg.norm(np.asarray(res["T_gripper_base_depth_cv"])[:3, 3] - np.asarray(truth["X_depth"])[:3, 3]) < 1e-3
    env_cfg, _ = load_config(site)                         # base: → default.yaml，再覆盖相机
    assert env_cfg.camera.intrinsics is not None and env_cfg.belt_center_x == EnvConfig().belt_center_x
    X_cfg = mujoco_cam_to_cv(env_cfg.camera.pos_in_gripper_base, env_cfg.camera.quat_in_gripper_base)
    assert np.allclose(X_cfg[:3, 3], np.asarray(truth["X_depth"])[:3, 3], atol=1e-3)
    v = step3_verify_in_sim.main(["--session", str(session), "--max-images", "1"])
    assert v["passed"] and v["mean_px"] < 1.0


def test_belt_measurement_recovers_scene_belt(q_home):
    cfg = EnvConfig()
    rig = SimRig(depth_fovy=75.0)
    # 深度相机在带中心线正上方、视线竖直朝下；绕竖直轴的朝向和高度挑一个够得着的
    tcp_in_gb = np.eye(4)
    tcp_in_gb[2, 3] = 0.1325
    q = None
    for h in (0.45, 0.40, 0.50):
        for yaw in np.radians(np.arange(0, 360, 30)):
            T_cam = np.eye(4)
            c, s = np.cos(yaw), np.sin(yaw)
            T_cam[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ np.diag([1.0, -1, -1])
            T_cam[:3, 3] = [cfg.belt_center_x, -0.10, cfg.belt_surface_z + h]
            T_tcp = T_cam @ he.inv(rig.X_true_depth) @ tcp_in_gb
            q_try, pe, re = rig.ik.solve(q_home, T_tcp[:3, 3], T_tcp[:3, :3], iters=300)
            if pe < 2e-3 and re < 2e-2:
                q = q_try
                break
        if q is not None:
            break
    assert q is not None, "找不到能让相机竖直朝下看带的关节角"
    rig.set_joints(q)
    _, depth = rig.render("depth")
    K = rig.intrinsics("depth").K
    pts = depth_to_base(depth, K, np.zeros(5), RobotFK().body_T(q) @ rig.X_true_depth)
    abc, on = fit_plane(pts, cfg.belt_surface_z)
    e = belt_edges(pts, on)
    assert abs(abc[2] - cfg.belt_surface_z) < 0.002
    assert abs(e["center_x"] - cfg.belt_center_x) < 0.006 and abs(e["half_width"] - cfg.belt_half_width) < 0.006
    assert e["edge_seen_low"] and e["edge_seen_high"]


def test_blind_zone_estimate():
    recs = [(d, 1.0) for d in np.linspace(0.40, 0.17, 50)] + [(0.0, 0.0)] * 10 + [(0.16, 0.3)] * 5
    est = estimate_min_depth(recs)
    assert 0.165 < est < 0.18
    assert estimate_min_depth([(0.3, 0.1)] * 20) is None


def test_multi_session_with_board_moved_between_sessions(q_home):
    """Board nudged between sessions: per-session board poses keep X exact; stability is reported."""
    rng = np.random.default_rng(3)
    board, fk = Chessboard(), RobotFK()
    X = mount_T(**DEFAULT_TRUE_MOUNT)
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    from nero_vs_rl.sim2real.calibration.board import solve_pnp
    samples = []
    for g in range(3):
        B = np.eye(4)
        B[:3, :3] = euler_xyz_to_R(0, 0, np.radians(1.5 * g)) @ np.diag([1.0, -1, -1])
        B[:3, 3] = [-0.50 + 0.008 * g, 0.08 - 0.005 * g, -0.116]
        rig = SimRig(board, T_base_board=B, X_true=X, seed=10 + g)
        for q in rig.random_view_joints(q_home, 8):
            T_cb = he.inv(X) @ he.inv(fk.body_T(q)) @ B
            uv, _ = cv2.projectPoints(board.object_points(), cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, np.zeros(5))
            corners = uv.reshape(-1, 2) + rng.normal(0, 0.3, (board.cols * board.rows, 2))
            T_pnp, rms = solve_pnp(corners, board, K, np.zeros(5))
            samples.append(he.Sample(fk.body_T(q), T_pnp, corners, rms, len(samples), group=g))
    rep = he.calibrate(samples, board, K, np.zeros(5))
    assert rep["n_groups"] == 3
    assert np.linalg.norm(rep["X"][:3, 3] - X[:3, 3]) < 1e-3
    assert rep["consistency"]["trans_max_m"] < 0.003          # 每个会话各自的板位姿 → 一致
    assert rep["stability"]["method"] == "leave_one_session_out" and rep["stability"]["trans_max_m"] < 0.002


def test_step2_merges_sessions(tmp_path):
    from nero_vs_rl.sim2real.calibration import step1_capture_handeye, step2_solve_handeye, step3_verify_in_sim
    a, b = tmp_path / "a_handeye", tmp_path / "b_handeye"
    step1_capture_handeye.main(["--sim", "--auto", "8", "--out", str(a)])
    step1_capture_handeye.main(["--sim", "--auto", "8", "--out", str(b)])
    res = step2_solve_handeye.main(["--session", str(a), str(b)])
    merged = Path(res["session"])
    assert merged.parent == tmp_path and merged.name.endswith("_handeye_merged")
    assert len(res["T_base_board_groups"]) == 2 and res["n_samples"] >= 14
    assert step3_verify_in_sim.main(["--session", str(merged), "--max-images", "1"])["passed"]


def test_step5_apply_from_saved_keeps_z(tmp_path):
    from nero_vs_rl.sim2real.calibration import step5_measure_belt
    from nero_vs_rl.sim2real.calibration.common import write_json
    dirs = []
    for i, cx in enumerate((-0.3773, -0.3775)):
        d = tmp_path / f"{i}_belt"
        d.mkdir()
        write_json(d / "belt.json", dict(center_x=cx, half_width=0.155, belt_surface_z_at_center=-0.112,
                                          reference_plane_z=-0.1224, edge_seen_low=True, edge_seen_high=True))
        dirs.append(str(d))
    site = tmp_path / "site.yaml"
    step5_measure_belt.main(["--from", *dirs, "--apply", "--keep-z", "--site-config", str(site)])
    env_cfg, _ = load_config(site)
    assert np.isclose(env_cfg.belt_center_x, -0.3774) and np.isclose(env_cfg.belt_half_width, 0.155)
    assert env_cfg.belt_surface_z == EnvConfig().belt_surface_z          # 高度没被改
