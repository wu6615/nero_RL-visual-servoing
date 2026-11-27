"""Eye-in-hand dynamic grasping task on the belt: GAP-RL "grasps as points" observations in MuJoCo.

Counterpart of GAP-RL ``envs/pick_single.py::PickSingleEnv``.
"""
from __future__ import annotations

from typing import Any

import dataclasses

import mujoco
import numpy as np
from gymnasium import spaces

from ..grasp_detection import analytic_grasps as gr
from ..sensors import camera as cam
from ..utils import object_library, traj_utils
from ..utils.geometry import expmap, inv_T, make_T, mat_to_quat_wxyz, rot_z, symmetric_grasp_rot_dist
from .base_env import BaseEnv
from .observation import GraspTracker, assemble_obs

PAD_GAP0_M = 0.008      # 指垫间距 = 0.008 + 2q（q = 单指行程，见 agents/robots/nero7.py）


# =============================================================================
# 【说明】动态抓取任务 —— 整个项目的"问题定义"都在这个类里
# =============================================================================
# 任务：NERO7 + 二指夹爪，用腕部 DC1（眼在手上）观察传送带上运动的物体，
#       在 max_steps（default.yaml：120 步 = 12 s）内抓住并抬起 lift_height_m 以上 → 成功。
#
# 坐标系：世界系 = NERO 基座系。带沿基座 -Y 运动（现场实测 belt_direction_base≈(-0.05,-1,0)）。
#
# 生命周期（父类 base_env.py 定义骨架，本类填充细节）：
#   __init__ ─→ BaseEnv.__init__ ─→ _load_scene（按配置摆带、改相机）→ _solve_home
#   reset    ─→ _initialize_episode：机器人初始位姿 → 选物体/尺寸 → 生成轨迹 → 解析抓取
#   step     ─→ 每个物理子步 _before_substep 驱动物体、_after_substep 查接触
#            ─→ get_obs ─→ evaluate ─→ compute_dense_reward
#
# 两套抓取，务必区分（与 GAP-RL 相同的思路）：
#   self.grasps            : 物体系下的"真值"抓取（grasp_detection/analytic_grasps.py 解析生成）
#                            → 只用于奖励（grasp_distances），属于特权信息，策略看不到
#   self._obs_grasps_ee    : 模拟感知后的抓取（EE 系）：只保留腕部相机可见的、加噪声、
#                            看不见时用记忆 + 速度外推 → 进入观测
#
# 物体运动：目标由 mocap 载体经 weld 约束沿轨迹驱动；机器人第一次碰到物体就关掉 weld
#           （同 GAP-RL _after_simulation_step 把驱动刚度清零），之后物体自由。
#           带面本身是一个沿 Y 滑动的刚体（_drive_belt）：每个子步把它的速度设成当前带速并拉回原点，
#           所以物体被碰掉、没抓稳掉回带上时，由真实接触摩擦带着走（会滑、会翻），和现场传送带一样。
# =============================================================================
class DynamicGraspEnv(BaseEnv):

    def __init__(self, config=None, render_mode: str | None = None):
        super().__init__(config, render_mode)
        g = self.cfg.grasp_obs
        self.N, self.K = g.num_grasps, g.num_points
        # 可夹持最大宽度 = 全开指垫间距 - 余量
        self.max_width = PAD_GAP0_M + 2 * self.cfg.gripper_max_q - g.finger_margin_m
        self._fixed_pts = (gr.gripper_keypoints(self.K) if g.points_mode == "keypoints"
                           else np.random.default_rng(1029).normal(0, gr.FINGER_HALF_OPEN / 3, (self.K, 3)))
        self.controller.set_workspace(
            [self.cfg.workspace_x[0], self.cfg.workspace_y[0], self.cfg.belt_surface_z + self.cfg.workspace_z_above_belt[0]],
            [self.cfg.workspace_x[1], self.cfg.workspace_y[1], self.cfg.belt_surface_z + self.cfg.workspace_z_above_belt[1]])

        # 【说明】动作 7 维见 agents/controllers/pd_ee_delta_pose.py。
        # 观测（与 GAP-RL state_egopoints 对应）：
        #   tcp_pose           (9)     TCP 位置 + 6D 旋转，基座系
        #   gripper_pos        (1)     夹爪开口，归一化
        #   action             (7)     上一步动作
        #   grasp_vis          (2)     [当前帧是否看到抓取, 记忆年龄]（GAP-RL 的 grasp_exist）
        #   origin_gripper_pts (K,3)   参考夹爪点 x_init（gauss 模式每步重采样）
        #   gripper_pts_diff   (N,K,3) 把 x_init 摆到每个观测抓取位姿上（EE 系），补零
        #   grasp_scores       (N)     抓取得分；0 表示该槽位为空（网络据此做 mask）
        inf = np.inf
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "tcp_pose": spaces.Box(-inf, inf, (9,), np.float32),
            "gripper_pos": spaces.Box(0.0, 1.0, (1,), np.float32),
            "action": spaces.Box(-1.0, 1.0, (7,), np.float32),
            "grasp_vis": spaces.Box(0.0, 1.0, (2,), np.float32),
            "origin_gripper_pts": spaces.Box(-inf, inf, (self.K, 3), np.float32),
            "gripper_pts_diff": spaces.Box(-inf, inf, (self.N, self.K, 3), np.float32),
            "grasp_scores": spaces.Box(0.0, 1.0, (self.N,), np.float32),
        })
        self._wrist = None
        self.tracker = GraspTracker(g, self.control_dt)     # 记忆 + 测速外推（真机共用，见 observation.py）

    # ------------------------------------------------------------------ scene
    def _load_scene(self):
        c, m = self.cfg, self.model
        self.belt_body, self.belt_geom = m.body("belt").id, m.geom("belt_surface").id
        self.rail_geoms = (m.geom("belt_rail_far").id, m.geom("belt_rail_near").id)
        self.obstacle_geoms = frozenset({self.belt_geom, *self.rail_geoms, m.geom("robot_pedestal").id})
        m.body_pos[self.belt_body] = [c.belt_center_x, 0.0, c.belt_surface_z - 0.02]
        m.geom_size[self.belt_geom] = [c.belt_half_width, c.belt_half_length, 0.02]
        jb = m.joint("belt_slide").id
        self.belt_qadr, self.belt_dadr = int(m.jnt_qposadr[jb]), int(m.jnt_dofadr[jb])
        # 【说明】两侧矮护栏（现场两侧都有，高约 1.5 cm）：世界系里的静止几何体，不随带面滑动；
        #   厚 2 cm，离带边 5 mm，顶面高出带面 belt_rail_height_m，往下延伸到带面下 2 cm
        half_h = (c.belt_rail_height_m + 0.02) / 2
        for side, rail in zip((-1.0, 1.0), self.rail_geoms):
            m.geom_pos[rail] = [c.belt_center_x + side * (c.belt_half_width + 0.015), 0.0,
                                c.belt_surface_z + c.belt_rail_height_m - half_h]
            m.geom_size[rail] = [0.010, c.belt_half_length, half_h]
        cc = c.camera
        # 渲染用竖直视场角；有标定内参时由 fy 换算，保证仿真画面与真机尺度一致
        m.cam_fovy[self.agent.camera] = (np.rad2deg(2 * np.arctan(cc.height / 2 / cc.intrinsics[1]))
                                         if cc.intrinsics is not None else cc.fovy_deg)
        if cc.pos_in_gripper_base is not None:
            m.cam_pos[self.agent.camera] = cc.pos_in_gripper_base
        if cc.quat_in_gripper_base is not None:
            m.cam_quat[self.agent.camera] = cc.quat_in_gripper_base
        if cc.pos_in_gripper_base is not None:            # 用了标定位姿：robot.xml 里占位的相机外壳不再对应真机，隐藏
            m.geom_rgba[m.geom("wrist_camera_housing").id, 3] = 0.0
        self.targets = {}
        for name, body, carrier, eq in (("box", "target_box", "box_carrier", "box_drive"),
                                        ("cylinder", "target_cyl", "cyl_carrier", "cyl_drive")):
            jnt = m.body(body).jntadr[0]
            self.targets[name] = dict(body=m.body(body).id, geom=m.body(body).geomadr[0],
                                      carrier=m.body(carrier).mocapid[0], eq=m.equality(eq).id,
                                      qadr=m.jnt_qposadr[jnt], dadr=m.jnt_dofadr[jnt])

    def _home_tcp_target(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        p = np.array([c.belt_center_x + c.init_tcp_offset_x, c.init_tcp_offset_y, c.belt_surface_z + c.init_tcp_height])
        # 与现场等待位一致：TCP z 朝下，y 沿 +Y（带向），x = y × z 指向 -X（远离机器人）
        R = np.column_stack([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        return p, R

    # ------------------------------------------------------------------ reset
    def _initialize_episode(self, options: dict | None):
        rng, d, c = self.np_random, self.data, self.cfg
        # 机器人：home TCP 附近随机平移 + 绕竖直轴随机偏航
        p0, R0 = self._home_tcp_target()
        p0 = p0 + rng.uniform(-1, 1, 3) * c.init_pos_noise_m
        R0 = rot_z(rng.uniform(-1, 1) * c.init_yaw_noise_rad) @ R0
        q0, _, _ = self.controller.ik.solve(self.q_home, p0, R0, iters=100)
        self.agent.set_state(d, q0, c.gripper_max_q)
        if c.finger_length_noise_m > 0:
            self.set_finger_length_offset(rng.uniform(-1, 1) * c.finger_length_noise_m)

        # 物体：随机类型/尺寸/质量 → 解析抓取；轨迹覆盖整个回合
        types = (options or {}).get("object_types", c.objects.types)
        self.lib_entry = None
        if c.objects.library:                             # 现场工件库：先抽工件，类型跟着工件走
            lib = object_library.load_library(c.objects.library, list(types))
            self.lib_entry = lib[int(rng.integers(len(lib)))]
            self.obj_type = self.lib_entry["type"]
        else:
            self.obj_type = str(rng.choice(types))
        self.obj = self.targets[self.obj_type]
        self._setup_object(rng)
        size = self.model.geom_size[self.obj["geom"]]
        half_x = (traj_utils.half_width_x_box(size[0], size[1]) if self.obj_type == "box"
                  else (lambda yaw, r=float(size[0]): r))
        self.traj = traj_utils.generate(rng, c, c.max_steps * self.control_dt + 1.0, half_x)
        self._attached = True
        for name, tgt in self.targets.items():
            d.eq_active[tgt["eq"]] = 1
            if tgt is not self.obj:                       # 没选中的物体停到远处
                park = np.array([2.0 + 0.5 * (name == "cylinder"), 2.0, -0.5])
                d.mocap_pos[tgt["carrier"]] = park
                d.qpos[tgt["qadr"]:tgt["qadr"] + 3] = park
        self._drive_target(0)
        q = d.qpos[self.obj["qadr"]:self.obj["qadr"] + 7]
        q[:3], q[3:] = d.mocap_pos[self.obj["carrier"]], d.mocap_quat[self.obj["carrier"]]

        # 感知记忆
        self.tracker.reset()
        self._contacts = dict(touch=False, left=False, right=False, collision=False)
        self._step_collision = False
        self._lifted_steps = 0

    def set_finger_length_offset(self, dl: float):
        """Move both finger collision meshes dl metres along the approach direction (+ = longer fingers).

        【说明】手指只沿闭合方向相对夹爪座平移，所以"伸出方向"在手指自身坐标系里是常量，第一次算好缓存。
        改的是 model.geom_pos（IK 私有 MjData 共用同一个 model），所以指尖抬高修正看到的也是改后的手指。
        """
        m, d = self.model, self.data
        if not hasattr(self, "_finger_axis"):
            mujoco.mj_kinematics(m, d)
            approach = d.site_xmat[m.site("ee_site").id].reshape(3, 3)[:, 2]
            self._finger_axis = {g: d.xmat[m.geom_bodyid[g]].reshape(3, 3).T @ approach for g in self.agent.finger_geoms}
            self._finger_pos0 = {g: m.geom_pos[g].copy() for g in self.agent.finger_geoms}
        for g in self.agent.finger_geoms:
            m.geom_pos[g] = self._finger_pos0[g] + self._finger_axis[g] * dl
        self.finger_length_offset = float(dl)

    def set_grip_friction(self, mu: float):
        """Effective finger-object sliding friction = mu (contacts take the larger of the two geoms, so set both).

        【说明】扭转、滚动摩擦按手指滑动系数同比例缩小（模型里手指 2.0）。物体和带面的接触仍由带面的 1.1 决定。
        """
        m, o = self.model, self.obj["geom"]
        if not hasattr(self, "_friction0"):
            self._friction0 = {g: m.geom_friction[g].copy() for g in self.agent.finger_geoms}
        for g in self.agent.finger_geoms:
            m.geom_friction[g] = self._friction0[g] * mu / self._friction0[g][0]
        m.geom_friction[o] = m.geom_friction[self.agent.finger_geoms[0]]

    def _setup_object(self, rng):
        m, oc, g = self.model, self.cfg.objects, self.cfg.grasp_obs
        gid, bid = self.obj["geom"], self.obj["body"]
        mass = rng.uniform(*oc.mass)
        # 运行时改几何尺寸：同时更新 rbound / aabb（碰撞粗筛用）和惯量
        lib_half = (None if self.lib_entry is None
                    else object_library.sample_half(self.lib_entry, rng, oc.library_size_noise))
        if self.obj_type == "box":
            half = (lib_half if lib_half is not None else
                    np.array([rng.uniform(*oc.box_half_x), rng.uniform(*oc.box_half_y), rng.uniform(*oc.box_half_z)]))
            m.geom_size[gid] = half
            m.geom_rbound[gid] = np.linalg.norm(half)
            m.geom_aabb[gid] = np.concatenate([np.zeros(3), half])
            inertia = mass / 3 * np.array([half[1]**2 + half[2]**2, half[0]**2 + half[2]**2, half[0]**2 + half[1]**2])
            self.obj_half_h = float(half[2])
            self.grasps = gr.box_grasps(half, self.max_width, g.max_tilt_deg)
        else:
            r, h = ((float(lib_half[0]), float(lib_half[2])) if lib_half is not None
                    else (rng.uniform(*oc.cyl_radius), rng.uniform(*oc.cyl_half_height)))
            m.geom_size[gid] = [r, h, 0]
            m.geom_rbound[gid] = np.hypot(r, h)
            m.geom_aabb[gid] = [0, 0, 0, r, r, h]
            inertia = mass * np.array([(3 * r**2 + 4 * h**2) / 12] * 2 + [r**2 / 2])
            self.obj_half_h = float(h)
            self.grasps = gr.cylinder_grasps(r, h, self.max_width, g.max_tilt_deg)
        if oc.grip_friction is not None:
            self.set_grip_friction(rng.uniform(*oc.grip_friction))
        m.body_mass[bid] = mass
        m.body_inertia[bid] = inertia
        m.body_ipos[bid] = 0
        m.body_iquat[bid] = [1, 0, 0, 0]
        # 改了质量/惯量必须重算派生常数（invweight0 等，影响约束软度）；不重算会在
        # 手指推挤物体时数值发散。mj_setConst 会改写传入的 MjData，所以用单独的草稿数据。
        if not hasattr(self, "_const_scratch"):
            self._const_scratch = mujoco.MjData(m)
        mujoco.mj_setConst(m, self._const_scratch)
        if len(self.grasps) == 0:
            raise RuntimeError("物体尺寸超出夹爪开口，没有可用抓取；检查 objects 配置")

    # -------------------------------------------------------- physics hooks
    def _drive_target(self, k: int):
        tr, d = self.traj, self.data
        i = min(k, len(tr.yaw) - 1)
        z = self.cfg.belt_surface_z + self.obj_half_h + 1e-4
        d.mocap_pos[self.obj["carrier"]] = [tr.xy[i, 0], tr.xy[i, 1], z]
        d.mocap_quat[self.obj["carrier"]] = mat_to_quat_wxyz(rot_z(tr.yaw[i]))

    def _drive_belt(self, k: int):
        # 【说明】带面速度 = 轨迹沿带方向（Y）的速度分量；位置每个子步拉回 0：
        #   带面是沿 Y 均匀的长条，拉回在几何上看不出来，但接触摩擦按速度算，物体照样被带走。
        #   贴着驱动的物体和带面同速，所以驱动阶段和以前一样没有相对滑动。
        d, i = self.data, min(k, len(self.traj.vel) - 1)
        d.qpos[self.belt_qadr] = 0.0
        d.qvel[self.belt_dadr] = self.traj.vel[i, 1]

    def _before_substep(self, k: int):
        self._drive_belt(k)
        if self._attached:
            self._drive_target(k)

    def _after_substep(self, k: int):
        self._contacts = self.agent.scan_contacts(self.data, self.obj["geom"], self.obstacle_geoms,
                                                  self.rail_geoms if self.cfg.finger_rail_collision else ())
        self._step_collision |= self._contacts["collision"]
        if self._contacts["touch"] and self._attached:
            self.data.eq_active[self.obj["eq"]] = 0      # 机器人一碰到物体就脱离驱动
            self._attached = False

    def _simulate(self, q_tgt, grip):
        self._step_collision = False
        super()._simulate(q_tgt, grip)

    # ------------------------------------------------------------ state utils
    def obj_T(self) -> np.ndarray:
        b = self.obj["body"]
        return make_T(self.data.xmat[b].reshape(3, 3), self.data.xpos[b])

    def true_grasps_world(self) -> np.ndarray:
        return np.einsum("ij,njk->nik", self.obj_T(), self.grasps.T_obj)

    # ------------------------------------------------------------ observation
    # 【说明】★ 模拟感知（真机上由 DC1 点云 + 抓取检测替代，见 sim2real/deploy/perception.py）：
    #   1. 每个抓取有一个"可见点"（物体顶面上方对应位置），投影到腕部相机：
    #      在画面内、深度在 [min_depth, max_depth]、射线不被遮挡 → 可见
    #      ⇒ 末端贴近物体时进入双目盲区，抓取会"消失"，这是眼在手上的核心难点
    #   2. 可见抓取加位置/姿态噪声；带噪物体位置送去测速
    #   3. 记忆、外推、观测拼装交给 observation.py（GraspTracker / assemble_obs），与真机同一份代码
    #   visible_rule = full_top（v6 起）：先要求物体顶面轮廓整个在画面内、深度在量程内，否则这一帧算"没看到"
    #   ——和真机感知一致（顶面被画面截断时尺寸量不准，perception.py 不用这一帧）。
    #   手指遮挡（2026-11-25 起真正生效）按顶面露出比例判：≥ min_top_exposed 才算看到；单个抓取不再判遮挡
    #   2026-11-27 起 default.yaml 取 min_top_exposed = 1.0（顶面完全露出）、并加上左侧双目无效带（camera.stereo_band_px_m）：
    #   真机感知改成"轮廓碰到大片无效深度或夹爪就不算看到"后，0.5 会让仿真多判 17% 的"看到"；1.0 一致率最高。
    #   （真机感知只要物体检出来，就给出全部抓取）。之前碰撞网格透明，mj_ray 跳过，遮挡一直没起作用。
    #   v5 用的 point 规则只要某个抓取的可见点在画面里就算看到，半截物体也有观测，部署时这部分帧会丢。
    def _observe_grasps(self) -> tuple[np.ndarray, np.ndarray, bool]:
        g, rng = self.cfg.grasp_obs, self.np_random
        T_wo = self.obj_T()
        camera = self.cfg.camera
        if g.visible_rule == "full_top":
            camera = dataclasses.replace(camera, occlusion_check=False)      # 轮廓和逐个抓取只看画面和深度；遮挡按顶面露出比例判
            top = (T_wo[:3, :3] @ self._top_outline().T).T + T_wo[:3, 3]
            see = lambda pts, c: cam.visible_mask(self.model, self.data, self.agent.camera, pts, self.obj["geom"], c)
            if not see(top, camera).all():
                return self.tracker.update(self.t, None)
            face = gr.top_face_samples(self.obj_type, self.model.geom_size[self.obj["geom"]])
            if see((T_wo[:3, :3] @ face.T).T + T_wo[:3, 3], self.cfg.camera).mean() < g.min_top_exposed:
                return self.tracker.update(self.t, None)
        elif g.visible_rule != "point":
            raise ValueError(f"未知 visible_rule: {g.visible_rule}")
        vis_world = (T_wo[:3, :3] @ self.grasps.vis_obj.T).T + T_wo[:3, 3]
        mask = cam.visible_mask(self.model, self.data, self.agent.camera, vis_world, self.obj["geom"], camera)
        if not mask.any():
            return self.tracker.update(self.t, None)
        Gw = self.true_grasps_world()[mask]
        Gw[:, :3, 3] += rng.normal(0, g.pos_noise_m, (len(Gw), 3))
        for j in range(len(Gw)):
            Gw[j, :3, :3] = expmap(rng.normal(0, np.deg2rad(g.rot_noise_deg), 3)) @ Gw[j, :3, :3]
        est = T_wo[:3, 3] + rng.normal(0, g.track_pos_noise_m, 3)
        return self.tracker.update(self.t, Gw, self.grasps.scores[mask], est)

    def _top_outline(self) -> np.ndarray:
        """Points on the top-face outline in the object frame (box: 4 corners, cylinder: 8 rim points), 2 mm inset."""
        size = self.model.geom_size[self.obj["geom"]]
        if self.obj_type == "box":
            hx, hy, hz = size - [0.002, 0.002, 0.0]
            return np.array([[sx * hx, sy * hy, hz] for sx in (-1, 1) for sy in (-1, 1)])
        r, hz = size[0] - 0.002, size[1]
        a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        return np.stack([r * np.cos(a), r * np.sin(a), np.full(8, hz)], axis=1)

    def get_obs(self) -> dict[str, np.ndarray]:
        p, R = self.agent.tcp_pose(self.data)
        Gw, scores, visible = self._observe_grasps()
        pts = gr.sample_gripper_points(self.cfg.grasp_obs.points_mode, self.K, self.np_random, self._fixed_pts)
        obs, self._obs_grasps_ee, self._obs_valid = assemble_obs(
            p, R, Gw, scores, visible, self.tracker.age_norm,
            self.agent.gripper_opening(self.data) / self.cfg.gripper_max_q, self.prev_action, pts, self.N)
        self._visible = visible
        return obs

    # ------------------------------------------------------- reward and eval
    # 【说明】末端到每个"真值抓取"的距离：平移 = EE 系下抓取原点模长；
    # 旋转 = 测地角 / π，且把绕接近轴转 180° 视为同一抓取（二指夹爪对称）。
    def grasp_distances(self) -> tuple[np.ndarray, np.ndarray]:
        p, R = self.agent.tcp_pose(self.data)
        Ge = np.einsum("ij,njk->nik", inv_T(make_T(R, p)), self.true_grasps_world())
        return np.linalg.norm(Ge[:, :3, 3], axis=1), symmetric_grasp_rot_dist(Ge[:, :3, :3]) / np.pi

    def jaw_state(self) -> dict[str, float]:
        """Jaw vs object right now: misalignment (rad), object width across the jaw, pad gap, tilt, relative speed."""
        d, m = self.data, self.model
        _, R = self.agent.tcp_pose(d)
        b, g = self.obj["body"], self.obj["geom"]
        misalign, width = gr.jaw_fit(self.obj_type, m.geom_size[g], d.xmat[b].reshape(3, 3).T @ R[:, 1])
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jac, None, self.agent.tcp_site)
        v_obj = d.qvel[self.obj["dadr"]:self.obj["dadr"] + 3]            # 自由关节前 3 维 = 世界系线速度
        return dict(misalign=misalign, width=width, gap=PAD_GAP0_M + 2 * self.agent.gripper_opening(d),
                    tilt=float(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))),
                    rel_speed=float(np.linalg.norm(jac @ d.qvel - v_obj)))

    def is_grasped(self, jaw: dict[str, float]) -> bool:
        """Both fingers on the object and (clean_grasp) squarely closed on it; also used by the virtual lab's truth."""
        c, ct = self.cfg, self._contacts
        if not (ct["left"] and ct["right"] and self.agent.gripper_opening(self.data) > 1e-3):
            return False
        return not c.clean_grasp or gr.is_clean_grasp(jaw["gap"], jaw["misalign"], jaw["width"],
                                                      c.clean_align_deg, c.clean_width_tol_m)

    def evaluate(self) -> dict[str, Any]:
        c, d = self.cfg, self.data
        jaw = self.jaw_state()
        grasped = self.is_grasped(jaw)
        obj_pos = d.xpos[self.obj["body"]]
        lift = float(obj_pos[2] - self.obj_half_h - c.belt_surface_z)
        lifted = grasped and lift > c.lift_height_m
        self._lifted_steps = self._lifted_steps + 1 if lifted else 0
        dt, dr = self.grasp_distances()
        gi = int(np.argmin(np.clip(dt, 0.01, 0.5) + np.clip(dr, 0.01, 0.5)))
        fell = bool(obj_pos[2] < c.belt_surface_z - 0.05)
        missed = bool(not grasped and obj_pos[1] * np.sign(c.belt_direction_sign) > abs(c.downstream_limit_y))
        success = bool(lifted and self._lifted_steps >= 2)
        return {
            "is_success": success, "success": success, "grasped": bool(grasped), "lifted": bool(lifted),
            "lift": lift, "robot_static": self.agent.arm_speed(d) < 0.2, "collision": bool(self._step_collision),
            "visible": bool(self._visible), "dist_t": float(dt[gi]), "dist_r": float(dr[gi]),
            "fell": fell, "missed": missed, "obj_type": self.obj_type,
            "jaw": jaw, "closing": bool(self.prev_action[6] < 0),
            "observed_grasps_ee": self._obs_grasps_ee.copy(), "observed_valid": self._obs_valid.copy(),
            "obj_vel_est": self.tracker.est_obj_vel.copy(),
        }

    # 【说明】★ 稠密奖励（结构沿用 GAP-RL compute_dense_reward，权重见 RewardConfig）：
    #   成功                       : success
    #   否则累加：
    #     info_exist   看得到抓取，或已贴近（<6 cm，进入盲区不应被惩罚）
    #     approach     接近最近的真值抓取，平移、旋转各 w·(1-tanh(5d))
    #     grasp        抓住（clean_grasp 打开时要真正夹住：对正且合到物体宽度，见 is_grasped）
    #     close_*      物体近旁要闭合但没夹住时，没对正 / 倾斜 / 手还在动（负，见 closing_penalty）
    #     lift         抓住后越接近抬升高度越高
    #     static       已抬起且机械臂基本静止
    #     collision    机械臂（非手指）碰到带面 / 护栏（负）
    #     time         时间惩罚（负）：成功之前每个控制步都扣，越快抓起来越好
    #   回合时长 max_steps（默认 120 步 = 12 s；11-22～11-24 是 80 步 = 8 s）内要完成"识别 → 抓取 → 抬起保持"，超时即失败。
    #   防钻空子：物体漏到下游 / 掉落会提前结束回合，此时把剩余步数的时间惩罚一次扣掉，
    #   否则策略会学到"看不见物体就放它流走、早结束少扣分"。
    #   成功后每步给 success（不再扣时间），所以越早成功，累计回报越高；成功后机械臂还在动再加 hold_still（负）。
    #   最后整体 × scale。真值抓取是特权信息，只在仿真训练中使用。
    def compute_dense_reward(self, info: dict[str, Any]) -> float:
        w = self.cfg.reward
        if info["success"]:
            # hold_still（v6 起）：成功后还在动就扣分——抓起来停住即可，不要带着物体继续走
            #   （v5 抬起后会把物体快速搬到工作空间角上；部署时也是抓起抬高后就停住）
            return (w.success + (0.0 if info["robot_static"] else w.hold_still)) * w.scale
        r = 0.0
        if info["visible"] or info["dist_t"] < 0.06:
            r += w.info_exist
        r += w.approach_trans * (1 - np.tanh(5 * np.clip(info["dist_t"], 0.01, 0.5)))
        r += w.approach_rot * (1 - np.tanh(5 * np.clip(info["dist_r"], 0.01, 0.5)))
        if info["grasped"]:
            r += w.grasp
            r += w.lift * (1 - np.tanh(5 * max(self.cfg.lift_height_m - info["lift"], 0.0)))
            if info["lifted"] and info["robot_static"]:
                r += w.static
        r += self.closing_penalty(info)
        if info["collision"]:
            r += w.collision
        r += w.time
        if info["missed"] or info["fell"]:
            r += w.time * max(self.cfg.max_steps - self.step_count, 0)
        return float(r * w.scale)

    def closing_penalty(self, info: dict[str, Any]) -> float:
        """Closing next to the object while askew, tilted or still moving relative to it (0 once cleanly grasped)."""
        w = self.cfg.reward
        if info["grasped"] or not info.get("closing") or info["dist_t"] >= w.close_near_m:
            return 0.0
        j = info["jaw"]
        return (w.close_misalign * min(j["misalign"] / np.deg2rad(45), 1.0)
                + w.close_tilt * float(np.clip((j["tilt"] - np.deg2rad(15)) / np.deg2rad(30), 0.0, 1.0))
                + w.close_moving * min(j["rel_speed"] / w.close_speed_ref, 1.0))

    def check_termination(self, info: dict[str, Any]) -> bool:
        return bool((info["success"] and self.cfg.terminate_on_success) or info["fell"] or info["missed"]
                    or (info["collision"] and self.cfg.terminate_on_collision))

    # ---------------------------------------------------------------- render
    def render_wrist_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB + depth (m, 0 = invalid/blind zone) from the wrist camera at DC1 resolution."""
        if self._wrist is None:
            self._wrist = cam.WristRGBD(self.model, self.cfg.camera)
        return self._wrist.render(self.data)

    def close(self):
        if self._wrist is not None:
            self._wrist.close()
            self._wrist = None
        super().close()
