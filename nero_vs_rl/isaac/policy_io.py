"""Move a policy between machines as plain tensors (no pickled Python objects tied to a Python / SB3 version)."""
from __future__ import annotations

from pathlib import Path

import torch

from ..utils.config import EnvConfig

# 【说明】SB3 的 .zip 里有按 Python 版本序列化的对象，服务器（3.11）存的在本机（3.10）可能读不了。
#   所以只搬策略网络的张量，在本机按同样的结构重建一个 SAC 再装进去，存成部署代码认识的 .zip。


def export_policy(model, path: Path, net_arch, N: int, K: int):
    torch.save({"state": {k: v.cpu() for k, v in model.policy.state_dict().items()}, "net_arch": list(net_arch), "N": N, "K": K,
                "num_timesteps": int(model.num_timesteps)}, path)


def import_policy(path: Path, out_zip: Path, cfg: EnvConfig):
    from gymnasium import Env
    from stable_baselines3 import SAC
    from ..algorithms.Networks import GraspPointGroupExtractor
    from .core import observation as O
    d = torch.load(path, map_location="cpu", weights_only=True)
    g = cfg.grasp_obs
    if (d["N"], d["K"]) != (g.num_grasps, g.num_points):
        raise ValueError(f"策略的抓取数/点数 {d['N']}/{d['K']} 和当前配置 {g.num_grasps}/{g.num_points} 不一致")

    N, K = d["N"], d["K"]

    class _Spaces(Env):                                   # 只提供空间，不会被步进
        observation_space, action_space = O.observation_space(N, K), O.action_space()

    model = SAC("MultiInputPolicy", _Spaces(), buffer_size=1, device="cpu",
                policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=d["net_arch"]))
    model.policy.load_state_dict(d["state"])
    model.num_timesteps = d["num_timesteps"]
    Path(out_zip).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_zip)
    return model
