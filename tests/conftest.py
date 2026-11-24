"""Shared fixtures; all tests are offline (no camera, CAN or GPU required)."""
import pytest

from nero_vs_rl.envs import DynamicGraspEnv
from nero_vs_rl.utils.config import EnvConfig


@pytest.fixture(scope="module")
def env():
    e = DynamicGraspEnv()
    yield e
    e.close()


@pytest.fixture
def make_env():
    made = []

    def _make(**overrides):
        e = DynamicGraspEnv(EnvConfig.from_dict(overrides))
        made.append(e)
        return e
    yield _make
    for e in made:
        e.close()
