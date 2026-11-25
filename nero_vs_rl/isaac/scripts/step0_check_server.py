"""Step 0 (server) - versions, GPU memory, disk, and that the files copied from the PC are complete. Does not start Isaac Sim.

  python -m nero_vs_rl.isaac.scripts.step0_check_server
"""
from __future__ import annotations

import importlib.metadata as im
import shutil
import sys

from ... import TRAIN_CONFIG
from .. import KINEMATICS_NPZ, OUTPUT_DIR, ROBOT_URDF

WANT = {"isaaclab": "2.3", "isaacsim": "5.1", "stable_baselines3": "2."}


def main():
    ok = True

    def item(good, text):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {text}")

    item(sys.version_info[:2] == (3, 11), f"Python {sys.version.split()[0]}（Isaac Sim 5.1 要 3.11）")
    for pkg, want in WANT.items():
        try:
            v = im.version(pkg)
        except im.PackageNotFoundError:
            v = None
        item(v is not None and (pkg == "isaaclab" or v.startswith(want)), f"{pkg} {v or '没装'}（要 {want}x）")
    import torch
    item(torch.cuda.is_available(), f"torch {torch.__version__}，CUDA {'可用' if torch.cuda.is_available() else '不可用'}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        item(p.total_memory > 15e9, f"{p.name}，显存 {p.total_memory / 1e9:.0f} GB（要 ≥ 16；A100 / H100 没有 RT Core，不支持）")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUTPUT_DIR).free / 1e9
    item(free > 20, f"输出目录 {OUTPUT_DIR}，剩余 {free:.0f} GB")
    for f in (KINEMATICS_NPZ, ROBOT_URDF, TRAIN_CONFIG):
        item(f.exists(), f"文件 {f}")
    item(TRAIN_CONFIG.name == "dc1_site.yaml", "用的是现场实测配置 dc1_site.yaml")
    n = len(list((ROBOT_URDF.parent / "../../assets/descriptions/nero7/meshes").resolve().glob("*.stl")))
    item(n == 12, f"机器人网格 {n} / 12 个")
    print("全部通过" if ok else "有没通过的项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
