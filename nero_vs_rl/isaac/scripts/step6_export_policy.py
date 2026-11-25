"""Step 6 (server) - write the policy weights as a plain tensor file to bring back to the PC. Does not start Isaac Sim.

  python -m nero_vs_rl.isaac.scripts.step6_export_policy --run-name isaac_v1
"""
from __future__ import annotations

import argparse

import yaml
from stable_baselines3 import SAC

from .. import OUTPUT_DIR
from ..policy_io import export_policy


def main(argv=None):
    ap = argparse.ArgumentParser(description="导出策略权重")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--which", choices=["best", "final", "paused"], default="best")
    args = ap.parse_args(argv)
    run = OUTPUT_DIR / args.run_name
    f = {"best": run / "best" / "best_model.zip", "final": run / "final_model.zip", "paused": run / "paused_model.zip"}[args.which]
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    g, out = cfg["task"]["grasp_obs"], run / f"policy_{args.which}.pt"
    export_policy(SAC.load(f, device="cpu", buffer_size=1), out, cfg["isaac"]["train"]["net_arch"], g["num_grasps"], g["num_points"])
    print(f"已导出 {out}，拷回本机后运行 step7_import_policy")


if __name__ == "__main__":
    main()
