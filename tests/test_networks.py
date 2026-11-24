"""Grasp point-group extractor: shapes, masking, and a 1-step SAC update (CPU)."""
import numpy as np
import torch

from nero_vs_rl.algorithms.Networks import GraspPointGroupEncoder, GraspPointGroupExtractor


def test_masked_slots_do_not_change_features():
    enc = GraspPointGroupEncoder()
    x0 = torch.randn(2, 16, 3)
    x = torch.randn(2, 20, 16, 3)
    valid = torch.zeros(2, 20, dtype=torch.bool)
    valid[:, :5] = True
    f1 = enc(x0, x, valid)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:]) * 10          # 改动无效槽位
    assert torch.allclose(f1, enc(x0, x2, valid))
    assert torch.all(enc(x0, x, torch.zeros_like(valid)) == 0)


def test_extractor_and_sac_update(env):
    from stable_baselines3 import SAC
    model = SAC("MultiInputPolicy", env, buffer_size=200, learning_starts=10, batch_size=8, device="cpu",
                policy_kwargs=dict(features_extractor_class=GraspPointGroupExtractor, net_arch=[32, 32]), seed=0)
    obs, _ = env.reset(seed=0)
    t = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
    feat = model.policy.actor.features_extractor(t)
    assert feat.shape == (1, 384)
    model.learn(total_timesteps=30)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (7,) and np.all(np.abs(action) <= 1)


def test_sac_train_warm_start_from_saved_model(tmp_path):
    """--init-model 读入另一次训练的权重：用默认（实测）配置，网络与保存的完全一致后再继续训。"""
    import torch
    import yaml
    from stable_baselines3 import SAC

    from nero_vs_rl import TRAIN_CONFIG
    from nero_vs_rl.algorithms.scripts import sac_train

    cfg = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    cfg.setdefault("train", {}).update(log_dir=str(tmp_path), batch_size=8, buffer_size=500, learning_starts=10_000)
    cfg["base"] = str(TRAIN_CONFIG.parent / cfg["base"])                              # 绝对路径，放到临时目录也能解析
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    sac_train.main(["--config", str(cfg_path), "--run-name", "a", "--total-timesteps", "20", "--n-envs", "1", "--no-eval"])
    src = tmp_path / "a" / "final_model.zip"
    # warm start 时 learning_starts 降到 1000 → 20 步内不会更新网络，权重应与读入的完全一致
    sac_train.main(["--config", str(cfg_path), "--run-name", "b", "--total-timesteps", "20", "--n-envs", "1",
                    "--no-eval", "--init-model", str(src)])
    a, b = SAC.load(src, device="cpu"), SAC.load(tmp_path / "b" / "final_model.zip", device="cpu")
    for (k, va), vb in zip(a.policy.state_dict().items(), b.policy.state_dict().values()):
        assert torch.equal(va, vb), k
    resolved = yaml.safe_load((tmp_path / "b" / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["init_model"] == str(src)
    assert resolved["env"]["camera"]["min_depth_m"] == cfg["env"]["camera"]["min_depth_m"]   # 用的是实测配置
