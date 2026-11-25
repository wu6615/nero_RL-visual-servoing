"""Start Isaac Sim with the shared command line options; import isaaclab modules only after calling start()."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def start(ap: argparse.ArgumentParser, argv=None):
    from isaaclab.app import AppLauncher
    from ... import TRAIN_CONFIG
    ap.add_argument("--task-config", default=str(TRAIN_CONFIG), help="任务配置（默认现场实测 dc1_site.yaml，和 MuJoCo 版同一份）")
    ap.add_argument("--isaac-config", default=None, help="Isaac 参数（默认 isaac/isaac.yaml）")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args(argv)
    app = AppLauncher(args).app
    from .. import ISAAC_CONFIG
    from ..config import load_isaac_config
    from ...utils.config import load_config
    task_cfg, _ = load_config(args.task_config)
    return args, app, task_cfg, load_isaac_config(Path(args.isaac_config) if args.isaac_config else ISAAC_CONFIG)


def finish(app, code: int):
    """Close the app and keep the real exit code (the app shutdown otherwise always returns 0)."""
    if sys.exc_info()[0] is not None:
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
    os._exit(code)
