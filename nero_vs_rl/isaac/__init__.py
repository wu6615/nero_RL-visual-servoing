"""Isaac Sim / Isaac Lab training side (GPU batch); deployment stays on MuJoCo kinematics (see README.md here)."""
import os
from pathlib import Path

ISAAC_DIR = Path(__file__).resolve().parent
KINEMATICS_NPZ = ISAAC_DIR / "assets" / "nero7_kinematics.npz"
ROBOT_URDF = ISAAC_DIR / "assets" / "nero7_isaac.urdf"
ISAAC_CONFIG = ISAAC_DIR / "isaac.yaml"
# 【说明】服务器上的输出目录（日志、模型、生成的 USD）。默认 ~/nero_runs，用环境变量 NERO_ISAAC_OUT 改
OUTPUT_DIR = Path(os.environ.get("NERO_ISAAC_OUT", str(Path.home() / "nero_runs")))
USD_DIR = OUTPUT_DIR / "usd"
