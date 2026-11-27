"""Workpiece library: STL fitting (synthetic meshes), the shipped library file, and env / Isaac / replay use of it."""
import dataclasses

import numpy as np
import pytest

from nero_vs_rl import TRAIN_CONFIG
from nero_vs_rl.utils import object_library as ol
from nero_vs_rl.utils.config import load_config


def _write_stl(path, tris):
    tris = np.asarray(tris, np.float32)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["v"] = tris
    path.write_bytes(b"\0" * 80 + len(tris).to_bytes(4, "little") + rec.tobytes())


def _prism(poly_xy, h):
    """Closed prism over a CCW polygon, outward-facing triangles (units: mm)."""
    P = np.asarray(poly_xy, float)
    c = P.mean(0)
    tris = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        tris += [[[*c, 0], [*b, 0], [*a, 0]], [[*c, h], [*a, h], [*b, h]],          # 底（朝下）、顶（朝上）
                 [[*a, 0], [*b, 0], [*b, h]], [[*a, 0], [*b, h], [*a, h]]]
    return np.array(tris)


def test_fit_box_and_cylinder_from_stl(tmp_path):
    _write_stl(tmp_path / "box.stl", _prism([[0, 0], [40, 0], [40, 70], [0, 70]], 30))
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    _write_stl(tmp_path / "cyl.stl", _prism(np.c_[25 * np.cos(a), 25 * np.sin(a)], 40))
    _write_stl(tmp_path / "wide.stl", _prism([[0, 0], [120, 0], [120, 150], [0, 150]], 20))
    lib = ol.build_library(tmp_path, max_width=0.096)
    got = {e["name"]: e for e in lib["objects"]}
    assert got["box"]["type"] == "box" and got["box"]["size_m"] == pytest.approx([0.04, 0.07, 0.03], abs=1e-4)
    assert got["box"]["volume_cm3"] == pytest.approx(84.0, rel=1e-3)
    assert got["cyl"]["type"] == "cylinder" and got["cyl"]["size_m"] == pytest.approx([0.05, 0.05, 0.04], abs=5e-4)
    assert [e["name"] for e in lib["excluded"]] == ["wide"]                         # 窄边 12 cm 夹不了


def test_shipped_library_is_graspable_and_used_by_default():
    cfg = load_config(TRAIN_CONFIG)[0]
    assert cfg.objects.library == "real_parts.yaml"
    lib = ol.load_library(cfg.objects.library)
    assert len(lib) == 15 and {e["type"] for e in lib} == {"box", "cylinder"}
    max_w = 0.008 + 2 * cfg.gripper_max_q - cfg.grasp_obs.finger_margin_m
    for e in lib:
        assert e["size_m"][0] * (1 + cfg.objects.library_size_noise) <= max_w + 1e-9   # 抖动后也夹得下
    oc = cfg.objects                                                                   # 尺寸范围盖住库（感知提示用）
    for e in lib:
        h = np.asarray(e["size_m"]) / 2
        if e["type"] == "box":
            assert oc.box_half_x[0] <= h[0] <= oc.box_half_x[1] and oc.box_half_y[0] <= h[1] <= oc.box_half_y[1]
            assert oc.box_half_z[0] <= h[2] <= oc.box_half_z[1]
        else:
            assert oc.cyl_radius[0] <= h[0] <= oc.cyl_radius[1] and oc.cyl_half_height[0] <= h[2] <= oc.cyl_half_height[1]


def test_env_samples_from_library_with_jitter():
    from nero_vs_rl.envs.dynamic_grasp import DynamicGraspEnv
    cfg = load_config(TRAIN_CONFIG)[0]
    env = DynamicGraspEnv(cfg)
    names = set()
    for seed in range(12):
        env.reset(seed=seed)
        e, size = env.lib_entry, env.model.geom_size[env.obj["geom"]]
        names.add(e["name"])
        assert env.obj_type == e["type"] and len(env.grasps) > 0
        want = np.asarray(e["size_m"]) / 2
        got = size if e["type"] == "box" else np.array([size[0], size[0], size[1]])
        assert np.all(np.abs(got / want - 1) <= cfg.objects.library_size_noise + 1e-9)
    assert len(names) >= 6
    env.reset(seed=0, options={"object_types": ["cylinder"]})
    assert env.lib_entry["type"] == "cylinder"
    env.close()


def test_isaac_specs_and_replay_world():
    from nero_vs_rl.isaac.core.objects import BOX, make_specs
    from nero_vs_rl.sim2real.deploy.reproduce_in_sim import world_config
    cfg = load_config(TRAIN_CONFIG)[0]
    lib = ol.load_library(cfg.objects.library)
    specs = make_specs(cfg, len(lib), seed=0)
    for s, e in zip(specs, lib):
        assert (s.kind == BOX) == (e["type"] == "box")
        assert np.all(np.abs(s.half / (np.asarray(e["size_m"]) / 2) - 1) <= 0.03 + 1e-9)
        assert 0.025 <= s.mass <= 0.05 and 0.45 <= s.friction <= 0.65
    w = world_config(cfg, dict(obj_type="box", half=np.array([0.03, 0.041, 0.0295]), center=np.array([-0.39, 0.12, -0.08]),
                               yaw=0.0, belt_z=-0.112), None)
    assert w.objects.library is None and w.objects.box_half_x == [0.03, 0.03]     # 复现指定尺寸，不用库
    assert dataclasses.replace(cfg.objects, library=None).library is None
