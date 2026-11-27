"""Partly seen objects (2026-11-27): perception rejects outlines touching large invalid depth (not holes inside the object); the sim has the stereo band."""
import dataclasses

import cv2
import numpy as np
import pytest

from nero_vs_rl import TRAIN_CONFIG
from nero_vs_rl.envs.dynamic_grasp import DynamicGraspEnv
from nero_vs_rl.sensors import camera as cam
from nero_vs_rl.utils.config import load_config
from nero_vs_rl.utils.geometry import make_T


@pytest.fixture(scope="module")
def cfg():
    return dataclasses.replace(load_config(TRAIN_CONFIG)[0], finger_length_noise_m=0.0)


@pytest.fixture(scope="module")
def seen_scene(cfg):
    """A start state where perception sees the whole object: (depth, detect(depth) callable, object pixel box)."""
    from nero_vs_rl.sim2real.deploy.common import RobotTwin
    from nero_vs_rl.sim2real.deploy.perception import BeltObjectDetector
    from nero_vs_rl.sim2real.hardware.sim_workcell import sim_depth_intrinsics
    env, tw, i = DynamicGraspEnv(cfg), RobotTwin(cfg), sim_depth_intrinsics(cfg)
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
    try:
        for seed in range(60):
            env.reset(seed=seed)
            depth = env.render_wrist_rgbd()[1]
            tw.set_joints(env.agent.arm_qpos(env.data))
            p, R = tw.tcp_pose()
            T_bc = tw.T_base_cam_cv()

            def detect(d, det=None):
                det = det or BeltObjectDetector(cfg, K, np.zeros(5), tw.max_width)
                return det.detect(d, T_bc, make_T(R, p)), det.debug.get("cut", "")

            got, cut = detect(depth)
            if got is not None and not got.truncated:
                # 物体顶面在图像里的像素框（用真值位姿投影）
                T = env.obj_T()
                top = (T[:3, :3] @ env._top_outline().T).T + T[:3, 3]
                pc = (np.linalg.inv(T_bc) @ np.c_[top, np.ones(len(top))].T).T[:, :3]
                uv = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), K, None)[0].reshape(-1, 2)
                return depth, detect, uv.min(0).astype(int), uv.max(0).astype(int)
        pytest.skip("前 60 个种子里没有起始就完整看到物体的")
    finally:
        env.close()


def test_whole_object_is_complete(seen_scene):
    depth, detect, lo, hi = seen_scene
    det, cut = detect(depth)
    assert det is not None and not det.truncated and cut == ""


def test_object_cut_by_invalid_depth_is_rejected(seen_scene):
    """Blank the left third of the object and everything left of it (like DC1's stereo band): not a whole object."""
    depth, detect, lo, hi = seen_scene
    d = depth.copy()
    d[:, : lo[0] + (hi[0] - lo[0]) // 3] = 0.0
    det, cut = detect(d)
    assert det is not None and det.truncated and cut.startswith("遮挡")


def test_narrow_shadow_next_to_object_is_not_an_occluder(seen_scene):
    """A 12 px invalid strip touching the object (stereo shadow width at 25-30 cm) must not reject it."""
    depth, detect, lo, hi = seen_scene
    d = depth.copy()
    d[lo[1]:hi[1], max(lo[0] - 12, 0):lo[0]] = 0.0
    det, cut = detect(d)
    assert det is not None and not det.truncated


def test_glare_hole_inside_object_is_not_an_occluder(seen_scene):
    """A 30 px invalid square in the middle of the top face (projector glare under the camera) leaves the outline whole."""
    depth, detect, lo, hi = seen_scene
    c = (lo + hi) // 2
    d = depth.copy()
    d[c[1] - 15:c[1] + 15, c[0] - 15:c[0] + 15] = 0.0
    det, cut = detect(d)
    assert det is not None and not det.truncated and cut == ""


def test_invalid_square_reaching_past_the_object_is_an_occluder(seen_scene):
    """The same square moved onto the object's left edge reaches outside the outline: the object may go on there."""
    depth, detect, lo, hi = seen_scene
    c = (lo + hi) // 2
    d = depth.copy()
    d[c[1] - 15:c[1] + 15, lo[0] - 15:lo[0] + 15] = 0.0
    det, cut = detect(d)
    assert det is not None and det.truncated and cut.startswith("遮挡")


def test_sim_camera_has_the_stereo_band(cfg):
    """Points left of u = stereo_band_px_m / depth are invisible; the rendered depth is empty there too."""
    assert cfg.camera.stereo_band_px_m == pytest.approx(19.0)
    env = DynamicGraspEnv(cfg)
    try:
        env.reset(seed=0)
        cid = env.agent.camera
        cp, cm = env.data.cam_xpos[cid].copy(), env.data.cam_xmat[cid].reshape(3, 3)
        K = cam.intrinsics(cfg.camera)
        z = 0.35
        pts = []
        for u in (40.0, 19.0 / z + cfg.camera.image_margin_px + 5):     # 带里 / 刚出带
            ray = np.array([(u - K[0, 2]) / K[0, 0], -(cfg.camera.height / 2 - K[1, 2]) / K[1, 1], -1.0]) * z
            pts.append(cp + cm @ ray)                                  # MuJoCo 相机系：x 右、y 上、看 −z
        vis = cam.visible_mask(env.model, env.data, cid, np.array(pts), -1, dataclasses.replace(cfg.camera, occlusion_check=False))
        assert list(vis) == [False, True]
        depth = env.render_wrist_rgbd()[1]
        rows = depth[150:350]
        u_first = np.argmax(rows > 0, axis=1)[(rows > 0).any(1)]
        z_first = rows[np.arange(len(rows))[(rows > 0).any(1)], u_first]
        assert np.all(u_first * z_first >= 19.0 - 0.5)                  # 每行第一个有深度的像素都在带外
    finally:
        env.close()
