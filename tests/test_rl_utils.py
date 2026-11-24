"""make_vec_env limits worker threads without changing the main (GPU) process environment."""
import os

from nero_vs_rl.algorithms.rl_utils import THREAD_ENV, make_vec_env
from nero_vs_rl.utils.config import EnvConfig


def test_thread_limit_does_not_leak_into_main_process(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    vec = make_vec_env(EnvConfig(), 2, 0)
    try:
        obs = vec.reset()
        assert obs["tcp_pose"].shape == (2, 9)
    finally:
        vec.close()
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ
    assert set(THREAD_ENV) >= {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
