"""Scene description for Isaac Lab: robot drives, belt, rails, pedestal, object library, contact sensors."""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg

from .. import USD_DIR
from ..config import IsaacConfig
from ..core.objects import BOX, ObjectSpec
from ...utils.config import EnvConfig

# =============================================================================
# 【说明】接口版本固定为 Isaac Lab 2.3.2（四元数 wxyz，数据是 torch 张量）。3.0 起四元数改 xyzw、数据要加 .torch，
#   要换版本时只改 sim/ 下的两个文件，core/ 不用动。
#   · 带面：和 MuJoCo 版同一个办法——一块很重、不受重力的刚体，每个物理步把速度设成带速、位置拉回原点，
#     碰掉的物体由摩擦带走。
#   · 物体：MultiAssetSpawnerCfg 按环境号依次取规格，要求 replicate_physics=False。
#   · 摩擦：物体材质 friction_combine_mode="min"，接触时取两者较小值 = 物体规格里的摩擦（手指、带面都更大）。
#   · 接触：在物体、带面、护栏、底座上各放一个接触传感器，按"机器人的哪个 body"过滤，一次拿到每个 body 的接触力。
# =============================================================================
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
GRIP_JOINTS = ["gripper_joint1", "gripper_joint2"]
FINGERS = ["gripper_link1", "gripper_link2"]
ARM_BODIES = [f"link{i}" for i in range(1, 8)] + ["gripper_flange", "gripper_base"]
BODIES = ARM_BODIES + FINGERS                              # 接触力矩阵的列顺序
ENV = "/World/envs/env_.*"          # 直接式环境里路径不会自动展开 {ENV_REGEX_NS}，写全
ROBOT = ENV + "/Robot"
USD_PATH = USD_DIR / "nero7.usd"
BELT_HALF_THICK = 0.02


def robot_cfg(ic: IsaacConfig, q_home: list[float], grip_open: float) -> ArticulationCfg:
    d = ic.drive
    per = lambda v: {f"joint{i + 1}": float(v[i]) for i in range(7)}
    return ArticulationCfg(
        prim_path=ROBOT,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),   # 重力由驱动器补偿
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, fix_root_link=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={**{f"joint{i + 1}": float(q_home[i]) for i in range(7)}, "gripper_joint1": grip_open, "gripper_joint2": -grip_open}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS, stiffness=per(d.stiffness), damping=per(d.damping),
                                       effort_limit_sim=d.effort, armature=d.armature, friction=d.joint_friction,
                                       dynamic_friction=d.joint_friction),
            "gripper": ImplicitActuatorCfg(joint_names_expr=GRIP_JOINTS, stiffness=d.gripper_stiffness, damping=d.gripper_damping,
                                           effort_limit_sim=d.gripper_effort, armature=0.001),
        })


def _material(mu: float, mode: str = "average"):
    return sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu, friction_combine_mode=mode)


def object_cfg(specs: list[ObjectSpec]) -> RigidObjectCfg:
    assets = []
    for s in specs:
        common = dict(mass_props=sim_utils.MassPropertiesCfg(mass=s.mass), physics_material=_material(s.friction, "min"),
                      rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, solver_position_iteration_count=12),
                      collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0), activate_contact_sensors=True)
        if s.kind == BOX:
            assets.append(sim_utils.CuboidCfg(size=tuple(float(2 * x) for x in s.half), **common))
        else:
            assets.append(sim_utils.CylinderCfg(radius=float(s.half[0]), height=float(2 * s.half[2]), **common))
    return RigidObjectCfg(prim_path=ENV + "/Object",
                          spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets, random_choice=False, activate_contact_sensors=True),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, 0.3, 0.2)))


def belt_cfg(c: EnvConfig, ic: IsaacConfig) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=ENV + "/Belt",
        spawn=sim_utils.CuboidCfg(
            size=(2 * c.belt_half_width, 2 * c.belt_half_length, 2 * BELT_HALF_THICK),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True), mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_material(ic.belt_friction), activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(c.belt_center_x, 0.0, c.belt_surface_z - BELT_HALF_THICK)))


def _static(name: str, spawn, pos) -> RigidObjectCfg:
    spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
    spawn.activate_contact_sensors = True
    return RigidObjectCfg(prim_path=f"{ENV}/{name}", spawn=spawn, init_state=RigidObjectCfg.InitialStateCfg(pos=pos))


def static_cfgs(c: EnvConfig) -> dict[str, RigidObjectCfg]:
    """Side rails and pedestal; same sizes as scene.xml / DynamicGraspEnv._load_scene."""
    half_h = (c.belt_rail_height_m + 0.02) / 2
    z = c.belt_surface_z + c.belt_rail_height_m - half_h
    out = {}
    for name, side in (("RailFar", -1.0), ("RailNear", 1.0)):
        out[name] = _static(name, sim_utils.CuboidCfg(size=(0.020, 2 * c.belt_half_length, 2 * half_h)),
                            (c.belt_center_x + side * (c.belt_half_width + 0.015), 0.0, z))
    out["Pedestal"] = _static("Pedestal", sim_utils.CylinderCfg(radius=0.085, height=0.588), (0.0, 0.0, -0.306))
    return out


def contact_cfg(prim: str, bodies: list[str]) -> ContactSensorCfg:
    return ContactSensorCfg(prim_path=f"{ENV}/{prim}", update_period=0.0, history_length=0,
                            filter_prim_paths_expr=[f"{ROBOT}/{b}" for b in bodies])
