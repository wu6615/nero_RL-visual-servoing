"""v8: a grasp counts only when the jaw is square on the object and really closed onto it; closing askew / moving is penalised."""
from types import SimpleNamespace

import numpy as np
import pytest

from nero_vs_rl import TRAIN_CONFIG
from nero_vs_rl.envs.dynamic_grasp import DynamicGraspEnv
from nero_vs_rl.grasp_detection import analytic_grasps as gr
from nero_vs_rl.utils.config import EnvConfig, load_config

BOX = np.array([0.030, 0.041, 0.0295])          # 2026-11-24 真机接线盒 6.0 × 8.2 × 5.9 cm


def _dir(deg):
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])


# ---------------------------------------------------------------- 以下不建仿真
def test_jaw_fit_box_and_cylinder():
    a, w = gr.jaw_fit("box", BOX, _dir(0))
    assert a == pytest.approx(0.0, abs=1e-6) and w == pytest.approx(0.060)
    a, w = gr.jaw_fit("box", BOX, -_dir(80))
    assert np.degrees(a) == pytest.approx(10.0) and w == pytest.approx(0.082)       # 最近的是 8.2 cm 那对面
    a, w = gr.jaw_fit("cylinder", np.array([0.03, 0.05, 0.0]), _dir(37))
    assert a == pytest.approx(0.0, abs=1e-6) and w == pytest.approx(0.06)           # 圆柱侧面任意方向都对正
    a, _ = gr.jaw_fit("cylinder", np.array([0.03, 0.05, 0.0]), np.array([np.cos(0.5), 0, np.sin(0.5)]))
    assert a == pytest.approx(0.5)


def test_clean_grasp_rejects_wedged_box():
    fit = lambda deg: gr.jaw_fit("box", BOX, _dir(deg))
    ok = lambda gap, deg: gr.is_clean_grasp(gap, *fit(deg), 20.0, 0.010)
    assert ok(0.061, 3) and ok(0.083, 88)
    assert not ok(0.108, 55)          # 仿真 032310：斜 55° 卡在全开的两指间（间距 10.8 cm）
    assert not ok(0.108, 88)          # 对正了但还没合上
    assert not ok(0.016, 3)           # 两指几乎合拢：物体不在中间
    assert not ok(0.061, 30)          # 没对正


def _info(**kw):
    jaw = dict(misalign=np.radians(50), tilt=np.radians(25), rel_speed=0.4, gap=0.1, width=0.08)
    jaw.update(kw.pop("jaw", {}))
    return dict(dict(grasped=False, closing=True, dist_t=0.03, jaw=jaw), **kw)


def test_closing_penalty_only_when_closing_near_and_not_grasped():
    cfg = load_config(TRAIN_CONFIG)[0]
    w = cfg.reward
    pen = lambda info: DynamicGraspEnv.closing_penalty(SimpleNamespace(cfg=cfg), info)
    assert pen(_info()) == pytest.approx(w.close_misalign + w.close_tilt / 3 + w.close_moving)   # 真机第一次闭合那种
    assert pen(_info(jaw=dict(misalign=0.0, tilt=0.1, rel_speed=0.0))) == pytest.approx(0.0)
    assert pen(_info(grasped=True)) == pen(_info(closing=False)) == pen(_info(dist_t=0.2)) == 0.0
    assert min(w.close_misalign, w.close_tilt, w.close_moving) < 0


def test_is_grasped_strict_only_when_enabled():
    agent = SimpleNamespace(gripper_opening=lambda d: 0.05)
    wedged = dict(misalign=np.radians(35), width=0.082, gap=0.108)
    for strict, want in ((False, True), (True, False)):
        e = SimpleNamespace(cfg=EnvConfig(clean_grasp=strict), _contacts=dict(left=True, right=True), agent=agent, data=None)
        assert DynamicGraspEnv.is_grasped(e, wedged) is want


def test_grip_friction_sets_finger_and_object():
    fr = np.array([[2.0, 0.025, 0.003], [2.0, 0.025, 0.003], [1.0, 0.02, 0.002]])
    e = SimpleNamespace(model=SimpleNamespace(geom_friction=fr), obj=dict(geom=2), agent=SimpleNamespace(finger_geoms=(0, 1)))
    for mu in (0.5, 0.8):                                   # 连续两回合：按原值算，不累乘
        DynamicGraspEnv.set_grip_friction(e, mu)
        assert np.maximum(fr[0], fr[2])[0] == pytest.approx(mu) and fr[1, 1] == pytest.approx(0.025 * mu / 2)


def test_old_configs_keep_old_behaviour():
    c = EnvConfig()
    assert not c.clean_grasp and c.objects.grip_friction is None
    assert c.reward.close_misalign == c.reward.close_tilt == c.reward.close_moving == 0.0
    c = load_config(TRAIN_CONFIG)[0]
    assert c.clean_grasp and c.objects.grip_friction == [0.45, 0.65]


# ---------------------------------------------------------------- 以下要建 MuJoCo 环境（不推进物理）
def test_jaw_state_at_reset():
    env = DynamicGraspEnv(load_config(TRAIN_CONFIG)[0])
    try:
        _, info = env.reset(seed=3)
        j = info["jaw"]
        assert j["gap"] == pytest.approx(0.108, abs=1e-3) and 0 <= j["misalign"] <= np.pi / 2
        assert j["tilt"] < np.radians(10) and not info["grasped"] and not info["closing"]
        g = env.agent.finger_geoms[0], env.obj["geom"]
        assert 0.4 <= max(env.model.geom_friction[x][0] for x in g) <= 0.9
    finally:
        env.close()
