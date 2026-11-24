"""Step 7 (lab, arm moves with --execute) - closed-loop grasping: the step6 workflow with the grasp profile.

  python -m nero_vs_rl.sim2real.deploy.step7_grasp                          # 只读预演：机械臂不动
  python -m nero_vs_rl.sim2real.deploy.step7_grasp --execute --episodes 3   # 先抓静止物体（带不开），再慢带，再提速

抓取档：速度同训练（降速会打乱策略的闭合时机，见 runtime/safety.py）、指尖不低于带面 8 mm（--finger-clearance-mm 可改）、
抓住并抬高 6 cm 后停住（不让策略带着物体继续走）。
"""
from __future__ import annotations

from .step6_follow import run_lab


def main(argv=None):
    return run_lab("grasp", argv)


if __name__ == "__main__":
    main()
