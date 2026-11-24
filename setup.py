"""Editable install for the nero_vs_rl package (same workflow as GAP-RL: pip install -e .)."""
from setuptools import find_packages, setup

setup(
    name="nero_vs_rl",
    version="0.1.0",
    description="RL + eye-in-hand visual servoing for dynamic grasping on AgileX NERO7",
    packages=find_packages(include=["nero_vs_rl", "nero_vs_rl.*"]),
    package_data={"nero_vs_rl": ["assets/descriptions/nero7/*.xml", "assets/descriptions/nero7/*.md",
                                 "assets/descriptions/nero7/LICENSE", "assets/descriptions/nero7/meshes/*.stl",
                                 "algorithms/config/*.yaml"]},
    python_requires=">=3.10",
)
