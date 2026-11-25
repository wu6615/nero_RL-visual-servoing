"""Fixed library of box/cylinder targets (sizes, mass, friction, analytic grasps) assigned to environments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...grasp_detection import analytic_grasps as gr
from ...utils.config import EnvConfig

# =============================================================================
# 【说明】抓取直接用现有的解析生成函数（analytic_grasps.py）逐个规格算好，再补零成批量张量，不另写一份。
#   每个规格：类型（0 = 长方体，1 = 圆柱）、半尺寸、质量、摩擦、抓取位姿/可见点/得分、顶面轮廓点。
#   环境 i 用规格 i % M；scene.py 生成 USD 物体时用同一个顺序（MultiAssetSpawnerCfg(random_choice=False)）。
# =============================================================================
BOX, CYL = 0, 1
PAD_GAP0_M = 0.008


@dataclass
class ObjectSpec:
    kind: int
    half: np.ndarray           # box: hx hy hz；cylinder: r r h
    mass: float
    friction: float


def make_specs(cfg: EnvConfig, n: int, seed: int) -> list[ObjectSpec]:
    rng, oc, out = np.random.default_rng(seed), cfg.objects, []
    fr = oc.grip_friction or [1.0, 1.0]
    for i in range(n):
        if oc.types[i % len(oc.types)] == "box":
            kind, half = BOX, np.array([rng.uniform(*oc.box_half_x), rng.uniform(*oc.box_half_y), rng.uniform(*oc.box_half_z)])
        else:
            r, h = rng.uniform(*oc.cyl_radius), rng.uniform(*oc.cyl_half_height)
            kind, half = CYL, np.array([r, r, h])
        out.append(ObjectSpec(kind, half, float(rng.uniform(*oc.mass)), float(rng.uniform(*fr))))
    return out


def top_outline(kind: int, half: np.ndarray) -> np.ndarray:
    """8 points on the top-face outline, 2 mm inset (box corners are repeated to 8)."""
    if kind == BOX:
        hx, hy, hz = half[0] - 0.002, half[1] - 0.002, half[2]
        return np.array([[sx * hx, sy * hy, hz] for sx in (-1, 1) for sy in (-1, 1)] * 2)
    a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    r = half[0] - 0.002
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(8, half[2])], 1)


class ObjectBatch:
    """Per-environment object tensors (row i = spec i % M)."""

    def __init__(self, specs: list[ObjectSpec], cfg: EnvConfig, num_envs: int, device="cpu", dtype=torch.float32):
        g = cfg.grasp_obs
        max_w = PAD_GAP0_M + 2 * cfg.gripper_max_q - g.finger_margin_m
        sets = [gr.box_grasps(s.half, max_w, g.max_tilt_deg) if s.kind == BOX
                else gr.cylinder_grasps(float(s.half[0]), float(s.half[2]), max_w, g.max_tilt_deg) for s in specs]
        if min(len(x) for x in sets) == 0:
            raise RuntimeError("有物体规格没有可用抓取；检查 objects 尺寸范围")
        G = max(len(x) for x in sets)
        T, vis = np.tile(np.eye(4), (len(specs), G, 1, 1)), np.zeros((len(specs), G, 3))
        sc, ok = np.zeros((len(specs), G)), np.zeros((len(specs), G), bool)
        for i, x in enumerate(sets):
            T[i, :len(x)], vis[i, :len(x)], sc[i, :len(x)], ok[i, :len(x)] = x.T_obj, x.vis_obj, x.scores, True
        idx = np.arange(num_envs) % len(specs)
        t = lambda a, d=dtype: torch.as_tensor(a[idx], device=device, dtype=d)
        self.kind = t(np.array([s.kind for s in specs]), torch.long)
        self.half = t(np.array([s.half for s in specs]))
        self.T_obj, self.vis_obj, self.scores, self.valid = t(T), t(vis), t(sc), t(ok, torch.bool)
        self.outline = t(np.array([top_outline(s.kind, s.half) for s in specs]))
        self.face = t(np.array([gr.top_face_samples("box", s.half) if s.kind == BOX
                                else gr.top_face_samples("cylinder", s.half[[0, 2]]) for s in specs]))
        self.G = G

    def half_width_x(self, yaw: torch.Tensor) -> torch.Tensor:
        """Half extent along base x on the belt (rows aligned with all envs)."""
        box = (self.half[:, 0] * torch.cos(yaw)).abs() + (self.half[:, 1] * torch.sin(yaw)).abs()
        return torch.where(self.kind == BOX, box, self.half[:, 0])

    def jaw_fit(self, close_dir_obj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analytic_grasps.jaw_fit: (misalignment rad, object width across the jaw m)."""
        c = close_dir_obj.abs()
        c = c / c.norm(dim=1, keepdim=True)
        cb, ib = c.max(1)
        wb = 2 * torch.gather(self.half, 1, ib[:, None]).squeeze(1)
        side, axial = c[:, :2].norm(dim=1), c[:, 2]
        cc = torch.maximum(side, axial)
        wc = torch.where(side >= axial, 2 * self.half[:, 0], 2 * self.half[:, 2])
        box = self.kind == BOX
        return torch.acos(torch.where(box, cb, cc).clamp(0, 1)), torch.where(box, wb, wc)
