"""SE(3) helper round-trips and grasp-symmetry distance."""
import numpy as np

from nero_vs_rl.utils.geometry import (RZ_PI, expmap, inv_T, logmap, make_T, mat_to_quat_wxyz, quat_wxyz_to_mat,
                                       rot_axis, symmetric_grasp_rot_dist)


def test_exp_log_roundtrip_including_near_pi():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.normal(size=3)
        w *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(w)
        assert np.allclose(logmap(expmap(w)), w, atol=1e-6)
    w = np.array([0, 0, np.pi - 1e-5])
    assert np.allclose(expmap(logmap(expmap(w))), expmap(w), atol=1e-6)


def test_quat_and_inverse():
    R = rot_axis([1, 2, 3], 0.7)
    assert np.allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(R)), R)
    T = make_T(R, np.array([0.1, -0.2, 0.3]))
    assert np.allclose(inv_T(T) @ T, np.eye(4))


def test_gripper_flip_is_zero_distance():
    assert np.isclose(symmetric_grasp_rot_dist(RZ_PI), 0.0)
    assert np.isclose(symmetric_grasp_rot_dist(rot_axis([1, 0, 0], 0.3)), 0.3)
