"""Batched NERO7 forward kinematics, Jacobian and gripper points from an exported table (no simulator needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .. import KINEMATICS_NPZ
from .so3 import expmap

# =============================================================================
# 【说明】TCP、相机、指尖高度都从关节角自己算，不读 Isaac 的 body 位姿：
#   · 真机部署用的就是"关节角 → 运动学"，两边口径一致；
#   · URDF → USD 转换后坐标系、body 名有出入也不影响策略看到的量（step2 会核对 Isaac 的 body 位姿）。
#   表由 isaac/export_assets.py 在本机从 MuJoCo 模型导出，服务器上不需要装 MuJoCo。
#   链：base → joint1..7（每个关节：固定变换 T_i，再绕局部 z 转 q_i）→ link7 → gripper_base → TCP。
#   夹爪顶点只存凸包（任意方向的最低点都在凸包上），分"夹爪座"和左右手指三组，都在 gripper_base 系。
# =============================================================================


class NeroKinematics:
    def __init__(self, path: Path = KINEMATICS_NPZ, device="cpu", dtype=torch.float32):
        z = np.load(path)
        t = lambda k: torch.as_tensor(z[k], device=device, dtype=dtype)
        self.R_fix, self.p_fix = t("R_fix"), t("p_fix")                # (7,3,3) (7,3)
        self.R_gb, self.p_gb = t("R_gb"), t("p_gb")                    # link7 → gripper_base
        self.p_tcp = t("p_tcp")                                        # gripper_base 系里的 TCP（姿态与 gripper_base 相同与否见 R_tcp）
        self.R_tcp = t("R_tcp")
        self.lo, self.hi = t("q_lo"), t("q_hi")
        self.verts_base, self.verts_f1, self.verts_f2 = t("verts_base"), t("verts_f1"), t("verts_f2")
        self.slide1, self.slide2 = t("slide1"), t("slide2")            # 手指每米行程在 gripper_base 系的位移方向
        self.planes_f1, self.planes_f2 = t("planes_f1"), t("planes_f2")  # 两指凸包的面 (F,4)：n·x + c ≤ 0 为内部，遮挡判断用
        self.device, self.dtype = device, dtype

    def joint_frames(self, q: torch.Tensor):
        """World rotation/position of every joint frame after its rotation, plus link7 frame. q (B,7)."""
        B = q.shape[0]
        R = torch.eye(3, device=q.device, dtype=q.dtype).expand(B, 3, 3)
        p = torch.zeros(B, 3, device=q.device, dtype=q.dtype)
        c, s = torch.cos(q), torch.sin(q)
        axes, origins = [], []
        for i in range(7):
            p = p + (R @ self.p_fix[i])
            R = R @ self.R_fix[i]
            axes.append(R[:, :, 2])
            origins.append(p)
            z = torch.zeros_like(c[:, i])
            o = torch.ones_like(z)
            Rz = torch.stack([c[:, i], -s[:, i], z, s[:, i], c[:, i], z, z, z, o], -1).reshape(B, 3, 3)
            R = R @ Rz
        return R, p, torch.stack(axes, 1), torch.stack(origins, 1)

    def gripper_base(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        R, p, _, _ = self.joint_frames(q)
        return R @ self.R_gb, p + R @ self.p_gb

    def tcp(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(p (B,3), R (B,3,3)) of the TCP in the robot base frame."""
        Rg, pg = self.gripper_base(q)
        return pg + Rg @ self.p_tcp, Rg @ self.R_tcp

    def tcp_jacobian(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(p, R, J (B,6,7)): linear rows first, world-aligned, like mj_jacSite."""
        R7, p7, axes, orig = self.joint_frames(q)
        Rg, pg = R7 @ self.R_gb, p7 + R7 @ self.p_gb
        p = pg + Rg @ self.p_tcp
        Jp = torch.cross(axes, p[:, None] - orig, dim=-1)
        return p, Rg @ self.R_tcp, torch.cat([Jp, axes], -1).transpose(1, 2)

    def gripper_lowest_z(self, q: torch.Tensor, grip_q: torch.Tensor) -> torch.Tensor:
        """Lowest z of the gripper meshes at opening grip_q (per-finger travel, m). (B,)"""
        Rg, pg = self.gripper_base(q)
        rz = Rg[:, 2]                                                   # 只要 z 分量：R 的第三行
        lows = [(rz @ self.verts_base.T).amin(1)]
        for v, sl in ((self.verts_f1, self.slide1), (self.verts_f2, self.slide2)):
            lows.append((rz @ v.T).amin(1) + (rz @ sl) * grip_q)
        return torch.stack(lows, 1).amin(1) + pg[:, 2]


def home_rotation(device, dtype) -> torch.Tensor:
    """Waiting-pose TCP orientation: z down, y along +Y (belt), same as DynamicGraspEnv._home_tcp_target."""
    return torch.tensor([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]], device=device, dtype=dtype)


__all__ = ["NeroKinematics", "home_rotation", "expmap"]
