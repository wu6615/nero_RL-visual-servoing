"""Isaac Lab DirectRLEnv: wires PhysX states into core/task.py and applies its targets."""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from ..config import IsaacConfig
from ..core.observation import action_space, observation_space
from ..core.so3 import yaw_to_quat_wxyz
from ..core.task import GraspTask, SimState
from ...utils.config import EnvConfig
from . import scene as S

FORCE_EPS = 1e-4            # 接触力超过这个值（N）算接触


@configclass
class NeroGraspEnvCfg(DirectRLEnvCfg):
    decimation: int = 20
    episode_length_s: float = 12.0
    observation_space = 1               # 占位，make_cfg 里换成字典空间
    action_space = 7
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=20)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=3.0, replicate_physics=False)


def make_env(task_cfg: EnvConfig, ic: IsaacConfig, num_envs: int | None = None, device: str = "cuda:0", seed: int = 0) -> "NeroGraspEnv":
    return NeroGraspEnv(make_cfg(task_cfg, ic, num_envs, device, seed), task_cfg, ic)


def make_cfg(task_cfg: EnvConfig, ic: IsaacConfig, num_envs: int | None = None, device: str = "cuda:0", seed: int = 0) -> NeroGraspEnvCfg:
    cfg = NeroGraspEnvCfg()
    ratio = 1.0 / (task_cfg.control_freq * ic.physics_dt)
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("控制周期必须是 physics_dt 的整数倍")
    cfg.decimation = int(round(ratio))
    cfg.sim.dt, cfg.sim.render_interval, cfg.sim.device = ic.physics_dt, cfg.decimation, device
    cfg.episode_length_s = task_cfg.max_steps / task_cfg.control_freq
    cfg.scene.num_envs, cfg.scene.env_spacing = num_envs or ic.num_envs, ic.env_spacing
    cfg.observation_space = observation_space(task_cfg.grasp_obs.num_grasps, task_cfg.grasp_obs.num_points)
    cfg.action_space = action_space()
    cfg.seed = seed
    return cfg


# =============================================================================
# 【说明】一个控制步（DirectRLEnv.step）：
#   _pre_physics_step  动作 → 关节目标（core/task.py 的批量控制器，和部署同一套算法）
#   _apply_action ×20  关节指令从上一目标线性插值到新目标；带面速度；没被碰到的物体贴着轨迹走；累计接触
#   _get_dones         任务 post_step：先按"重置之前"的状态算观测、奖励、终止（终止观测给 SB3 用）
#   _get_rewards / _reset_idx / _get_observations
#   接触用每个物理步的接触力，控制步内取"或"：只看最后一个物理步会漏掉短促的碰撞。
# =============================================================================
class NeroGraspEnv(DirectRLEnv):
    cfg: NeroGraspEnvCfg

    def __init__(self, cfg: NeroGraspEnvCfg, task_cfg: EnvConfig, isaac_cfg: IsaacConfig, render_mode=None, **kw):
        self.task_cfg, self.isaac_cfg = task_cfg, isaac_cfg
        # GraspTask 要先建：_setup_scene 需要物体规格和 home 关节
        self.task = GraspTask(task_cfg, cfg.scene.num_envs, isaac_cfg.object_library, isaac_cfg.object_seed,
                              seed=cfg.seed or 0, device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kw)
        r = self.robot
        self.arm_ids, names = r.find_joints(S.ARM_JOINTS, preserve_order=True)
        self.grip_ids, gnames = r.find_joints(S.GRIP_JOINTS, preserve_order=True)
        if names != S.ARM_JOINTS or gnames != S.GRIP_JOINTS:
            raise RuntimeError(f"USD 里的关节名和预期不一致：{r.joint_names}")
        B, dev = self.num_envs, self.device
        self.q_prev = self.task.ctl.q_cmd.clone()
        self.q_tgt, self.grip_tgt = self.q_prev.clone(), torch.full((B,), task_cfg.gripper_max_q, device=dev)
        self.sub = 0
        flag = lambda: torch.zeros(B, dtype=torch.bool, device=dev)
        self.touch, self.left, self.right, self.coll = flag(), flag(), flag(), flag()
        self.belt_pose0 = torch.zeros(B, 7, device=dev)
        self.belt_pose0[:, 0], self.belt_pose0[:, 2] = task_cfg.belt_center_x, task_cfg.belt_surface_z - S.BELT_HALF_THICK
        self.belt_pose0[:, 3] = 1.0
        self.belt_pose0[:, :3] += self.scene.env_origins
        self.finger_cols = [S.BODIES.index(b) for b in S.FINGERS]
        self.arm_cols = [S.BODIES.index(b) for b in S.ARM_BODIES]
        self.terminal_obs, self.task_info, self.fresh = None, {}, None

    # ------------------------------------------------------------ scene
    def _setup_scene(self):
        c, ic = self.task_cfg, self.isaac_cfg
        self.robot = Articulation(S.robot_cfg(ic, self.task.q_home.tolist(), c.gripper_max_q))
        self.object = RigidObject(S.object_cfg(self.task.specs))
        self.belt = RigidObject(S.belt_cfg(c, ic))
        statics = {k: RigidObject(v) for k, v in S.static_cfgs(c).items()}
        # 【说明】replicate_physics=False 时各环境的根节点在 InteractiveScene 里已经建好，上面的路径通配会逐个环境生成；
        #   这里不能再调 clone_environments，否则会用 0 号环境的物体覆盖其他规格。只需要隔开环境间的碰撞。
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects.update({"object": self.object, "belt": self.belt, **{k.lower(): v for k, v in statics.items()}})
        self.sensors = {k: ContactSensor(S.contact_cfg(k, S.BODIES)) for k in ("Object", "Belt", "RailFar", "RailNear", "Pedestal")}
        self.scene.sensors.update({f"contact_{k.lower()}": v for k, v in self.sensors.items()})
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------ state
    def _hits(self, name: str) -> torch.Tensor:
        """(B, len(BODIES)) bool: which robot bodies press on this prim right now."""
        f = self.sensors[name].data.force_matrix_w
        return f.reshape(self.num_envs, -1, len(S.BODIES), 3).norm(dim=-1).amax(1) > FORCE_EPS

    def _scan_contacts(self):
        o = self._hits("Object")
        self.left, self.right = o[:, self.finger_cols[0]], o[:, self.finger_cols[1]]          # 抓住与否看最后一个物理步
        self.touch |= o.any(1)
        arm = torch.zeros_like(self.coll)
        for k in ("Belt", "RailFar", "RailNear", "Pedestal"):
            h = self._hits(k)
            arm |= h[:, self.arm_cols].any(1)
            if k.startswith("Rail") and self.task_cfg.finger_rail_collision:
                arm |= h[:, self.finger_cols].any(1)
        self.coll |= arm
        self.task.note_touch(o.any(1))

    def _state(self) -> SimState:
        d, o = self.robot.data, self.object.data
        g = d.joint_pos[:, self.grip_ids]
        return SimState(q=d.joint_pos[:, self.arm_ids], qd=d.joint_vel[:, self.arm_ids], grip=(g[:, 0] - g[:, 1]) / 2,
                        obj_pos=o.root_link_pos_w - self.scene.env_origins, obj_quat=o.root_link_quat_w,
                        obj_vel=o.root_link_lin_vel_w, left=self.left, right=self.right, touch=self.touch, collision=self.coll)

    # ------------------------------------------------------------ stepping
    def _pre_physics_step(self, actions: torch.Tensor):
        self.q_prev, self.q_tgt, self.grip_tgt = self.task.pre_step(self._state(), actions)
        self.sub = 0
        self.touch, self.coll = torch.zeros_like(self.touch), torch.zeros_like(self.coll)

    def _apply_action(self):
        if self.sub > 0:
            self._scan_contacts()                                     # 上一个物理步的接触
        self.sub += 1
        w = self.sub / self.cfg.decimation
        self.robot.set_joint_position_target(self.q_prev + (self.q_tgt - self.q_prev) * w, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(torch.stack([self.grip_tgt, -self.grip_tgt], 1), joint_ids=self.grip_ids)
        self._drive()

    def _drive(self):
        """Belt surface motion, and objects that have not been touched yet follow their trajectory."""
        pos, yaw, vel = self.task.object_drive()                      # 这一物理步开始时应在的位姿
        bv = torch.zeros(self.num_envs, 6, device=self.device)
        bv[:, 1] = vel[:, 1]
        self.belt.write_root_pose_to_sim(self.belt_pose0)
        self.belt.write_root_velocity_to_sim(bv)
        ids = self.task.attached.nonzero().squeeze(-1)
        if len(ids):
            self._place_object(ids, pos, yaw, vel)
        self.task.substep(self.cfg.sim.dt)

    def _place_object(self, ids, pos, yaw, vel):
        pose = torch.cat([pos + self.scene.env_origins, yaw_to_quat_wxyz(yaw)], 1)[ids]
        self.object.write_root_pose_to_sim(pose, env_ids=ids)
        self.object.write_root_velocity_to_sim(torch.cat([vel, torch.zeros_like(vel)], 1)[ids], env_ids=ids)

    def _get_dones(self):
        self._scan_contacts()                                         # 最后一个物理步
        obs, self.task_reward, term, trunc, self.task_info = self.task.post_step(self._state())
        bad = ~torch.isfinite(self.task_reward)                       # 数值发散：不计奖励，直接结束
        self.task_reward = torch.where(bad, torch.zeros_like(self.task_reward), self.task_reward)
        self.terminal_obs = obs
        self.task_info["sim_unstable"] = bad
        return term | bad, trunc & ~bad

    def _get_rewards(self):
        return self.task_reward

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        super()._reset_idx(ids)
        q0 = self.task.reset(ids)
        g = self.task_cfg.gripper_max_q
        joint_ids = list(self.arm_ids) + list(self.grip_ids)
        grip = torch.full((len(ids), 1), g, device=self.device)
        q = torch.cat([q0, grip, -grip], 1)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=joint_ids, env_ids=ids)
        self.robot.set_joint_position_target(q, joint_ids=joint_ids, env_ids=ids)
        self._place_object(ids, *self.task.object_drive())
        if hasattr(self, "q_tgt"):                                    # 第一次 reset 在 __init__ 之后才有这些缓冲
            self.q_prev[ids], self.q_tgt[ids], self.grip_tgt[ids] = q0, q0, g
            for f in (self.touch, self.left, self.right, self.coll):
                f[ids] = False
        self.fresh = ids

    def _get_observations(self):
        if self.terminal_obs is None or (self.fresh is not None and len(self.fresh) == self.num_envs):
            obs = self.task.observe(self._state(), self.task.all_ids)
        else:
            obs = {k: v.clone() for k, v in self.terminal_obs.items()}
            if self.fresh is not None:
                for k, v in self.task.observe(self._state(), self.fresh).items():
                    obs[k][self.fresh] = v
        self.fresh = None
        return {"policy": obs}
