"""Analytic grasp frames and belt trajectories."""
import numpy as np

from nero_vs_rl.grasp_detection.analytic_grasps import TIP_AHEAD_OF_TCP, box_grasps, cylinder_grasps, gripper_keypoints
from nero_vs_rl.utils import traj_utils
from nero_vs_rl.utils.config import EnvConfig


def test_box_grasps_are_valid_rotations_closing_on_narrow_side():
    half = np.array([0.02, 0.06, 0.03])
    gs = box_grasps(half, max_width=0.096, max_tilt_deg=15)
    assert len(gs) > 0
    for T in gs.T_obj:
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
        assert R[2, 2] < -0.9                                # 接近方向朝下
        assert abs(R[0, 1]) > 0.99                           # 只沿 12 cm 长边的垂直方向（x）闭合
        tip = T[:3, 3] + TIP_AHEAD_OF_TCP * R[:, 2]
        assert tip[2] >= -half[2]                            # 指尖不进带面


def test_too_wide_objects_get_no_grasps():
    assert len(box_grasps(np.array([0.06, 0.06, 0.02]), 0.096, 0)) == 0
    assert len(cylinder_grasps(0.06, 0.03, 0.096, 0)) == 0
    assert len(cylinder_grasps(0.02, 0.03, 0.096, 15)) == 12


def test_keypoints_shape():
    assert gripper_keypoints(16).shape == (16, 3)


def test_line_trajectory_moves_downstream_on_belt():
    cfg = EnvConfig()
    tr = traj_utils.generate(np.random.default_rng(1), cfg, 10.0)
    assert np.all(np.diff(tr.xy[:, 1]) <= 1e-12)             # 沿 -Y
    assert np.all(np.abs(tr.xy[:, 0] - cfg.belt_center_x) <= cfg.belt_half_width)
    speed = np.linalg.norm(tr.vel, axis=1)
    assert speed.min() > 0 and speed.max() <= cfg.trajectory.speed[1] + 1e-12


def test_varying_speed_stays_inside_configured_range():
    """vary_speed adds ±10–35 %, but the belt speed must never leave its range: trajectory.speed for normal
    episodes, trajectory.slow_speed for slow ones (v7), exactly 0 for static ones."""
    from nero_vs_rl import TRAIN_CONFIG
    from nero_vs_rl.utils.config import load_config
    cfg, _ = load_config(TRAIN_CONFIG)
    lo, hi = cfg.trajectory.speed
    slo, shi = cfg.trajectory.slow_speed
    assert (lo, hi) == (0.04, 0.50)
    seen = []
    for s in range(80):
        sp = np.linalg.norm(traj_utils.generate(np.random.default_rng(s), cfg, 10.0).vel, axis=1)
        if sp.max() == 0.0:
            continue                                               # 静止回合
        band = (lo, hi) if sp.min() >= lo - 1e-12 else (slo, shi)
        assert sp.min() >= band[0] - 1e-12 and sp.max() <= band[1] + 1e-12
        seen.append(sp.mean())
    assert min(seen) < 0.04 and max(seen) > 0.35                 # 慢带和高速档都抽得到


def test_bezier_trajectory_stays_on_belt():
    cfg = EnvConfig.from_dict({"trajectory": {"mode": "bezier2d"}})
    for s in range(5):
        tr = traj_utils.generate(np.random.default_rng(s), cfg, 10.0)
        assert np.all(np.abs(tr.xy[:, 0] - cfg.belt_center_x) <= cfg.belt_half_width)
