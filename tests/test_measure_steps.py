"""Steps 8-10 (lab measurements without a policy): helpers offline, plus short dry runs in the MuJoCo cell."""
import numpy as np
import pytest

from nero_vs_rl.sim2real.deploy import step8_dynamics_check as s8
from nero_vs_rl.sim2real.deploy import step9_gripper_check as s9
from nero_vs_rl.sim2real.deploy import step10_grasp_probe as s10
from nero_vs_rl.sim2real.deploy.runtime.scripted import best_lag
from nero_vs_rl.utils.geometry import rot_z

BOX = dict(obj_type="box", half=[0.030, 0.041, 0.0295], center=[-0.39, 0.12, -0.0845], yaw=0.0)


def test_best_lag_recovers_a_known_delay():
    t = np.arange(0, 2, 0.01)
    cmd = np.sin(3 * t)
    assert best_lag(t, cmd, t, np.sin(3 * (t - 0.12)))[0] == pytest.approx(0.12, abs=0.006)


def test_gripper_timing_numbers():
    ts = np.arange(0, 1.0, 0.01)
    ws = np.clip(0.10 - (ts - 0.03) * 0.5, 0.0, 0.10)              # 30 ms 后开始，0.2 s 合完
    r = s9.timing(ts, ws)
    assert r["start_s"] == pytest.approx(0.04, abs=0.011) and r["half_s"] == pytest.approx(0.13, abs=0.011)
    assert r["done_s"] == pytest.approx(0.23, abs=0.011) and r["to_m"] == 0.0


def test_dynamics_plan_returns_to_start():
    moves = s8.plan()
    assert np.allclose(sum(d for _, d, _ in moves), 0) and max(v for _, _, v in moves) <= 0.5


def test_probe_geometry():
    from nero_vs_rl import TRAIN_CONFIG
    from nero_vs_rl.sim2real.deploy.perception import BeltObjectDetector
    from nero_vs_rl.utils.config import load_config
    cfg = load_config(TRAIN_CONFIG)[0]
    det = BeltObjectDetector(cfg, np.eye(3), np.zeros(5), 0.096)
    R_home = np.diag([-1.0, 1.0, -1.0])
    G = s10.base_grasp(BOX, det, R_home)
    assert abs(G[0, 1]) > 0.99                                     # 闭合方向（TCP y）跨 6 cm 窄边 = 基座 x
    assert np.allclose(G[:2, 3], BOX["center"][:2], atol=1e-6) and G[2, 2] < -0.99
    p, R = s10.trial_pose(G, 0.01, 0.02, 15)
    assert np.allclose(R.T @ (p - G[:3, 3]), rot_z(np.radians(15)).T @ [0.02, 0.01, 0.0])
    assert not s10.finger_on_top(BOX, *s10.trial_pose(G, 0.0, 0.0, 0))
    assert not s10.finger_on_top(BOX, *s10.trial_pose(G, 0.01, 0.0, 0))
    assert s10.finger_on_top(BOX, *s10.trial_pose(G, 0.015, 0.0, 0))          # 每侧只剩约 1.4 cm
    assert s10.finger_on_top(BOX, *s10.trial_pose(G, 0.0, 0.0, 45))           # 斜 45° 指尖落在顶面角上
    assert not s10.finger_on_top(BOX, *s10.trial_pose(G, 0.0, 0.0, 90))       # 90° = 跨 8.2 cm 长边，全开 10.8 cm 放得下


def test_step8_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(s8, "SPEEDS", (0.2,))
    monkeypatch.setattr(s8, "AXES", s8.AXES[:1])
    rep = s8.main(["--sim", "--no-viewer", "--out-root", str(tmp_path)])
    assert not rep["problems"] and len(rep["segments"]) == 4
    for r in rep["segments"]:
        assert 0.1 < r["lag_s"] < 0.35 and r["final_err_m"] < 0.003
    assert (next(tmp_path.glob("*_dynamics_check")) / "joints.npz").exists()


def test_step9_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(s9, "TRIALS", 1)
    rep = s9.main(["--sim", "--no-viewer", "--out-root", str(tmp_path)])
    close = next(r for r in rep["timing"] if r["kind"] == "闭合")
    assert close["from_m"] > 0.095 and close["to_m"] < 0.005 and 0 < close["done_s"] < 0.5
    assert all(abs(r["gap_m"] - r["reading_m"] - 0.008) < 1e-6 for r in rep["ruler"])


def test_step10_dry_run_and_replay(tmp_path):
    rep = s10.main(["--sim", "--no-viewer", "--trials", "0,3", "--out-root", str(tmp_path / "a")])
    first, skipped = rep["trials"]
    assert first["held"] and first["truth_lifted"] and "skipped" in skipped
    run = next((tmp_path / "a").glob("*_grasp_probe"))
    again = s10.main(["--sim", "--no-viewer", "--replay-real", str(run), "--out-root", str(tmp_path / "b")])
    assert [t["spec"] for t in again["trials"]] == [t["spec"] for t in rep["trials"]] and again["trials"][0]["held"]
