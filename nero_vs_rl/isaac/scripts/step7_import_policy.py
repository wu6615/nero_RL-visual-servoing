"""Step 7 (development PC) - rebuild an SB3 .zip that step2 replay / step4 dry run / the lab programs can load.

  python -m nero_vs_rl.isaac.scripts.step7_import_policy --pt /home/yunfeiwu/adddisk1/MY_RL_test/simulation/isaac_v1/policy_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ... import TRAIN_CONFIG
from ...utils.config import load_config
from ..policy_io import import_policy


def main(argv=None):
    ap = argparse.ArgumentParser(description="把服务器导出的策略转成部署用的 .zip")
    ap.add_argument("--pt", required=True)
    args = ap.parse_args(argv)
    out = Path(args.pt).with_suffix(".zip")
    import_policy(Path(args.pt), out, load_config(TRAIN_CONFIG)[0])
    print(f"已生成 {out}\n验收（部署路径干跑）：\n"
          f"python -m nero_vs_rl.sim2real.deploy.step4_closed_loop_sim --profile grasp --episodes 100 --no-viewer --lab-belt --model {out}")


if __name__ == "__main__":
    main()
