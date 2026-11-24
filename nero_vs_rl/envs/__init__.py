"""Gymnasium registration for the NERO eye-in-hand dynamic grasp env."""
from gymnasium.envs.registration import register

from .base_env import BaseEnv
from .dynamic_grasp import DynamicGraspEnv

register(id="NeroDynamicGrasp-v0", entry_point="nero_vs_rl.envs.dynamic_grasp:DynamicGraspEnv")

__all__ = ["BaseEnv", "DynamicGraspEnv"]
