"""Typed configuration for the eye-in-hand dynamic grasp env and SAC training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    # 占位值：DC1 的真实安装位姿和内参装好后用手眼标定 / 出厂内参替换。
    # None 表示沿用 robot.xml 里 wrist_camera 的位姿（相对 gripper_base）。
    pos_in_gripper_base: list[float] | None = None
    quat_in_gripper_base: list[float] | None = None   # wxyz
    fovy_deg: float = 75.0            # 占位值（robot.xml 原值），待用 DC1 深度内参替换
    width: int = 640                  # DC1 深度工厂模式 640x400（旧 dabai.py 后端）
    height: int = 400
    # 标定后的精确针孔内参 [fx, fy, cx, cy]（像素，对应 width x height）。给出时可见性投影用它，
    # MuJoCo 渲染仍只能用 fovy（由 fy 换算），主点偏移在渲染里忽略。
    intrinsics: list[float] | None = None
    min_depth_m: float = 0.15         # 占位值：双目近距盲区，需现场实测
    max_depth_m: float = 1.2
    image_margin_px: int = 8
    occlusion_check: bool = True
    # 真机：相机帧比关节反馈晚多少秒（帧的曝光时刻 ≈ 到达时刻 − latency_s），部署时按这个时刻取关节角。
    # 由 sim2real/deploy/step3_measure_latency.py --apply 写入；仿真不用。
    latency_s: float = 0.0


@dataclass
class ObjectConfig:
    types: list[str] = field(default_factory=lambda: ["box", "cylinder"])
    box_half_x: list[float] = field(default_factory=lambda: [0.015, 0.035])   # 可夹持方向
    box_half_y: list[float] = field(default_factory=lambda: [0.020, 0.060])
    box_half_z: list[float] = field(default_factory=lambda: [0.015, 0.040])
    cyl_radius: list[float] = field(default_factory=lambda: [0.015, 0.035])
    cyl_half_height: list[float] = field(default_factory=lambda: [0.015, 0.050])
    mass: list[float] = field(default_factory=lambda: [0.05, 0.30])


# 现场带速范围：2026-11-22 部署离线回放测得三档 3.6 / 9.6 / 13.3 cm/s（docs/现场几何来源.md），留余量取 15 cm/s。
# 只用于评估时单独统计"现场带速"成功率，不影响训练分布（训练范围见 trajectory.speed）。
SITE_BELT_SPEED_MAX = 0.15


@dataclass
class TrajectoryConfig:
    mode: str = "line"                 # line | bezier2d | static
    speed: list[float] = field(default_factory=lambda: [0.02, 0.08])   # m/s
    vary_speed: bool = True            # 段内平滑变速（GAP-RL vary_speed）
    lateral: list[float] = field(default_factory=lambda: [-0.07, 0.07])  # 相对带中心线（x 方向）
    # True（v6 起）：横向位置在整条带宽上均匀随机，按物体大小和朝向收进护栏内（离护栏 ≥ 5 mm），忽略 lateral
    lateral_full_belt: bool = False
    start_along: list[float] = field(default_factory=lambda: [0.12, 0.32])  # 起点 y（基座系，上游为 +Y）
    yaw_rate_max: float = 0.0          # rad/s，line 模式物体自转；bezier2d 会沿切向转
    # v7 起：真机测试多是静止 / 很慢的物体，训练最低 4 cm/s 没见过。每回合先按 static_prob 抽"静止"，
    #   否则按 slow_prob 抽"慢速"（带速取 slow_speed 范围），剩下的照 speed。都为 0 时不抽随机数（旧配置可复现）
    static_prob: float = 0.0
    slow_prob: float = 0.0
    slow_speed: list[float] = field(default_factory=lambda: [0.005, 0.04])


@dataclass
class GraspObsConfig:
    num_grasps: int = 20               # 观测中最多 N 个抓取（不足补零）
    num_points: int = 16               # 每个抓取 K 个点
    points_mode: str = "gauss"         # gauss | gauss_fix | keypoints
    pos_noise_m: float = 0.004
    rot_noise_deg: float = 3.0
    memory_mode: str = "extrapolate"   # none | hold | extrapolate
    memory_max_age_steps: int = 30
    track_pos_noise_m: float = 0.002   # 跟踪得到的物体位置噪声（用于测速）
    vel_window: int = 8                # 测速：最近 W 个可见帧最小二乘求斜率
    finger_margin_m: float = 0.012     # 物体宽度 + 余量 <= 最大张开
    max_tilt_deg: float = 15.0
    # 什么时候算"这一帧看到了"：
    #   point     每个抓取的可见点各自判断（v5 及以前；看到半截物体也有抓取）
    #   full_top  物体顶面整个在画面内、深度在量程内、没被挡住才算看到，然后再逐个判断抓取
    #             （和真机感知一致：顶面被画面边缘截断时尺寸量不准，deploy/perception.py 不用这一帧）
    visible_rule: str = "point"


@dataclass
class RewardConfig:
    success: float = 15.0
    info_exist: float = 1.0
    approach_trans: float = 3.0
    approach_rot: float = 3.0
    grasp: float = 3.0
    lift: float = 2.0
    static: float = 1.0
    collision: float = -1.0
    time: float = -1.0            # 成功前每个控制步的时间惩罚（× scale）；提前失败结束时补扣剩余步数
    hold_still: float = 0.0       # 成功之后机械臂还在动（关节最大速度 ≥ 0.2 rad/s）时每步加这一项（负数 = 惩罚）
    scale: float = 0.5


@dataclass
class EnvConfig:
    control_freq: float = 10.0
    sim_timestep: float = 0.002
    max_steps: int = 100
    action_frame: str = "ee"           # ee | base
    max_delta_pos_m: float = 0.02      # 每控制步最大平移
    max_delta_rot_rad: float = 0.10
    # 关节角速度上限（rad/s），可给一个数（所有关节相同）或 7 个数（逐关节）
    max_joint_speed_rad_s: float | list[float] = 1.5
    gripper_max_q: float = 0.05        # gripper_joint1 行程；指垫间距 = 0.008 + 2q
    # 带的位姿（基座系）。来源见 docs/现场几何来源.md：
    #   belt_surface_z  实测带面平面 c 项（belt_direction_20261113_single_pass.json）
    #   belt_center_x   估计值：现场等待位 TCP 的 x（standby_joint_rad 正解），带宽未实测
    belt_center_x: float = -0.40
    belt_surface_z: float = -0.12
    belt_half_width: float = 0.15      # 占位：带宽未实测
    belt_half_length: float = 1.0
    belt_direction_sign: float = -1.0  # 物体沿 sign*Y 运动；现场 belt_direction_base≈(-0.05,-1,0)
    belt_rail_height_m: float = 0.015  # 两侧护栏高出带面的高度（2026-11-22 用户现场说明：两侧都有，约 1.5 cm）
    init_tcp_height: float = 0.22      # 初始 TCP 离带面高度（现场等待位约 0.15）
    init_tcp_offset_y: float = 0.07    # 现场等待位 TCP 的 y
    # 初始 TCP 相对带中心线的 x 偏移（负 = 远离机器人）。腕部相机装在夹爪靠机器人一侧，视野中心偏向机器人，
    # TCP 正对带中心时远侧 7 cm 看不到；物体横向铺满整条带（lateral_full_belt）时要往外挪，让视野盖住整条带
    init_tcp_offset_x: float = 0.0
    init_pos_noise_m: float = 0.03
    init_yaw_noise_rad: float = 0.4
    # 现场等待位关节（two_second.hardware.tracked_freeze.yaml standby_joint_rad），作为 home IK 种子
    home_seed_joint_rad: list[float] = field(default_factory=lambda: [
        -0.15616595258058463, 0.6563590912851938, -0.08736010956498129, 1.6638224933253727,
        -0.08130801279383741, -0.293856030555404, 0.6642143667591787])
    workspace_x: list[float] = field(default_factory=lambda: [-0.62, -0.18])
    workspace_y: list[float] = field(default_factory=lambda: [-0.30, 0.45])  # 现场跟随区间 y∈[-0.2,0.38] 加余量
    workspace_z_above_belt: list[float] = field(default_factory=lambda: [0.008, 0.40])
    # 目标修正（训练和部署共用，见 agents/controllers/pd_ee_delta_pose.py 的 correct）：
    #   finger_clearance_m      夹爪最低点至少高出带面这么多，不够就把目标竖直抬高；None = 不管（v5 及以前）
    #   joint_limit_margin_rad  关节目标截到 [下限 + 此值, 上限 − 此值]；0 = 只截到硬限位
    #   workspace_pullback      IK 解偏出工作空间时朝工作空间再解一次
    finger_clearance_m: float | None = None
    joint_limit_margin_rad: float = 0.0
    workspace_pullback: bool = False
    # 手指碰到护栏也算碰撞（扣 reward.collision）。v6 起打开：物体可以贴着护栏走，部署时任何部件碰护栏都会停
    finger_rail_collision: bool = False
    # v7 起：每回合手指（碰撞体）沿伸出方向随机伸缩 ±此值，模拟手指实际长度和名义尺寸的偏差
    #   （仿真复现里手指缩短 1 cm 会明显降低第一次闭合就夹住的比例）。部署孪生和虚拟实验室不随机（用名义尺寸）
    finger_length_noise_m: float = 0.0
    lift_height_m: float = 0.06
    downstream_limit_y: float = -0.25
    terminate_on_collision: bool = False
    # 同 GAP-RL get_done：成功后不提前结束，继续拿成功奖励直到回合结束 → "抓住并保持"才最优；
    # 若成功即结束，策略反而可能靠在物体旁徘徊累积稠密奖励。失败（掉落/漏过/数值发散）仍提前结束。
    terminate_on_success: bool = False
    camera: CameraConfig = field(default_factory=CameraConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    grasp_obs: GraspObsConfig = field(default_factory=GraspObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EnvConfig":
        return _build(cls, data or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    seed: int = 0
    device: str = "auto"
    learning_rate: float = 3e-4
    buffer_size: int = 100_000         # 15 GB 内存机器上 N=20,K=16 约 0.8 GB
    batch_size: int = 256
    gamma: float = 0.95
    tau: float = 0.005
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    checkpoint_every: int = 100_000
    eval_every: int = 50_000
    eval_episodes: int = 20
    log_dir: str = "/home/yunfeiwu/adddisk1/MY_RL_test/simulation"   # 数据盘；见 nero_vs_rl.SIM_OUTPUT_DIR

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainConfig":
        return _build(cls, data or {})


def _build(cls, data: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__} 不认识的配置项: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if hasattr(default, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Read a YAML config; ``base: other.yaml`` (relative to this file) is loaded first and overridden.

    标定结果写成 ``base: default.yaml`` + 只含标定值的小文件（见 sim2real/calibration/sim_bridge.py），
    这样默认配置以后改了，标定文件不用跟着改。``provenance`` 节只做记录，不参与配置。
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - {"env", "train", "base", "provenance"}
    if unknown:
        raise ValueError(f"配置文件只允许 env / train / base / provenance，多出: {sorted(unknown)}")
    base = data.pop("base", None)
    data.pop("provenance", None)
    if base:
        data = _deep_merge(load_config_dict(path.parent / base), data)
    return data


def load_config(path: str | Path) -> tuple[EnvConfig, TrainConfig]:
    data = load_config_dict(path)
    return EnvConfig.from_dict(data.get("env")), TrainConfig.from_dict(data.get("train"))
