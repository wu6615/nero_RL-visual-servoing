"""Step 1 (server) - URDF -> USD.

  python -m nero_vs_rl.isaac.scripts.step1_convert_robot --headless
"""
from __future__ import annotations

import argparse
import sys

from ._app import finish, start


def main():
    args, app, _, ic = start(argparse.ArgumentParser(description="NERO7 URDF 转 USD"))
    code = 1
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from .. import ROBOT_URDF, USD_DIR
        from ..sim.scene import USD_PATH
        # 【说明】merge_fixed_joints=False：保留 gripper_flange / gripper_base，接触过滤和 step2 核对要用这些 body 名。
        #   驱动参数这里随便给，真正的值在 sim/scene.py 的执行器里覆盖。
        gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=50.0, damping=5.0)
        conv = UrdfConverter(UrdfConverterCfg(
            asset_path=str(ROBOT_URDF), usd_dir=str(USD_DIR), usd_file_name=USD_PATH.name, fix_base=True, merge_fixed_joints=False,
            force_usd_conversion=True, collider_type="convex_hull", self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(gains=gains, target_type="position")))
        print(f"已生成 {conv.usd_path}")
        code = 0
    finally:
        finish(app, code)


if __name__ == "__main__":
    sys.exit(main())
