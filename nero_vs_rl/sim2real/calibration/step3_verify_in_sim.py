"""Step 3 - check the calibration *inside the simulation env*: same joints, does the sim camera see the board where the real one did?

  python -m nero_vs_rl.sim2real.calibration.step3_verify_in_sim --session calib_data/<目录>

输出 <session>/verify/：每张样本一幅"真机图（绿=检测角点，红=仿真预测）| 仿真腕部相机渲染"对照图。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from ...envs.dynamic_grasp import DynamicGraspEnv
from ...sensors import camera as cam
from ...utils.config import EnvConfig
from .board import Chessboard
from .common import read_json, read_jsonl, write_json
from .handeye import rot_deg
from .sim_bridge import camera_overrides

# =============================================================================
# 【说明】这一步验证的是"标定结果放进仿真之后是否正确"，而不是重复 step2 的数学：
#   1. 用 step2 的彩色相机外参/内参生成 env.camera 覆盖项（走 sim_bridge 的 OpenCV→MuJoCo 转换）
#   2. 建一个真正的 DynamicGraspEnv（就是训练用的那个类），把关节设成每张样本的实测关节角
#   3. 用仿真相机位姿（data.cam_xpos / cam_xmat）把"标定板在基座系的位姿 B"上的角点投影到图像
#   4. 与真机图上检测到的角点比较（先对检测角点去畸变，因为仿真相机是理想针孔）
#   判定：仿真算出的每张样本误差，应与 step2 用 OpenCV 算的整链重投影误差一致（相差 < 0.5 px）。
#   注意误差本身不会是 0：它包含机械臂名义运动学的绝对误差（真机上几个像素），这在 step2 已经评估过；
#   这里只验证"放进仿真"这一步没有引入新的偏差（坐标系转换、父 body、配置加载）。
#   另外核对深度相机：仿真里深度相机的位姿应等于 彩色位姿 · 出厂深度→彩色外参。
# =============================================================================


def env_with_camera(overrides: dict) -> DynamicGraspEnv:
    return DynamicGraspEnv(EnvConfig.from_dict({"camera": overrides}))


def sim_camera_T(env: DynamicGraspEnv, q: np.ndarray) -> np.ndarray:
    """OpenCV-frame camera pose in base after setting the arm to q (gripper open, as during capture)."""
    env.agent.set_state(env.data, q, env.cfg.gripper_max_q)
    mujoco.mj_forward(env.model, env.data)
    T = np.eye(4)
    T[:3, :3] = env.data.cam_xmat[env.agent.camera].reshape(3, 3) @ np.diag([1.0, -1, -1])
    T[:3, 3] = env.data.cam_xpos[env.agent.camera]
    return T


def render_sim(env: DynamicGraspEnv) -> np.ndarray:
    """Wrist camera RGB, scaled down to fit MuJoCo's offscreen buffer (1280x800) if needed."""
    w, h = env.cfg.camera.width, env.cfg.camera.height
    s = min(1.0, env.model.vis.global_.offwidth / w, env.model.vis.global_.offheight / h)
    r = mujoco.Renderer(env.model, height=int(h * s), width=int(w * s))
    r.update_scene(env.data, camera=env.agent.WRIST_CAMERA)
    img = r.render().copy()
    r.close()
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description="手眼标定第 3 步：在仿真环境里核对标定结果")
    ap.add_argument("--session", required=True)
    ap.add_argument("--max-images", type=int, default=12, help="最多保存几张对照图")
    args = ap.parse_args(argv)
    session = Path(args.session)
    res = read_json(session / "result.json")
    board = Chessboard(**res["board"])
    board_pts = board.object_points()
    ci = res["color_intrinsics"]
    K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1.0]])
    D = np.asarray(ci["dist"])
    X_color = np.asarray(res["T_gripper_base_color_cv"])
    B_groups = {int(g): np.asarray(B) for g, B in (res.get("T_base_board_groups") or {"0": res["T_base_board"]}).items()}
    pts_base_g = {g: (B[:3, :3] @ board_pts.T).T + B[:3, 3] for g, B in B_groups.items()}   # 每个会话一个板位姿

    env = env_with_camera(camera_overrides(X_color, ci))
    cc = env.cfg.camera
    out = session / "verify"
    out.mkdir(exist_ok=True)
    errs, saved, diffs = [], 0, []
    used = set(res["sample_ids"])
    ref = dict(zip(res["sample_ids"], res["reprojection_px"]))     # step2 的整链重投影 RMS（OpenCV 计算）
    for r in read_jsonl(session / "samples.jsonl"):
        if r["id"] not in used:
            continue
        sim_camera_T(env, np.asarray(r["joints"]))
        uv, _ = cam.project(pts_base_g[int(r.get("group", 0))], env.data.cam_xpos[env.agent.camera].copy(),
                            env.data.cam_xmat[env.agent.camera].reshape(3, 3), cc)
        det = np.asarray(r["corners"], float)
        det_ud = cv2.undistortPoints(det.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        e = np.linalg.norm(uv - det_ud, axis=1)
        errs.append(float(np.sqrt(np.mean(e ** 2))))
        diffs.append(abs(errs[-1] - ref[r["id"]]))
        if saved < args.max_images:
            real = imageio.imread(session / r["image"])        # 合并会话里是绝对路径，Path 拼接后仍正确
            vis = real.copy()
            for p in det:
                cv2.circle(vis, tuple(int(v) for v in p), 4, (0, 255, 0), -1)
            for p in uv:
                cv2.circle(vis, tuple(int(v) for v in p), 6, (255, 0, 0), 1)
            rgb = cv2.resize(render_sim(env), (vis.shape[1], vis.shape[0]))   # 仿真场景里没有标定板：看手指、带面位置
            imageio.imwrite(out / f"{r['id']:03d}_real_vs_sim.jpg", np.concatenate([vis, rgb], axis=1))
            saved += 1
    env.close()

    depth_check = None
    if res.get("T_gripper_base_depth_cv") is not None and res.get("depth_intrinsics"):
        X_depth = np.asarray(res["T_gripper_base_depth_cv"])
        env_d = env_with_camera(camera_overrides(X_depth, res["depth_intrinsics"]))
        q = np.asarray(read_jsonl(session / "samples.jsonl")[0]["joints"])
        T_c = sim_camera_T(env_with_camera(camera_overrides(X_color, ci)), q)
        T_d = sim_camera_T(env_d, q)
        expect = T_c @ np.asarray(res["T_color_depth"])
        depth_check = dict(trans_err_m=float(np.linalg.norm(T_d[:3, 3] - expect[:3, 3])),
                           rot_err_deg=rot_deg(T_d[:3, :3].T @ expect[:3, :3]))
        env_d.close()

    mean, worst, max_diff = float(np.mean(errs)), float(np.max(errs)), float(np.max(diffs))
    ok = max_diff < 0.5 and (depth_check is None or depth_check["trans_err_m"] < 1e-4)
    write_json(out / "verify.json", dict(per_sample_rms_px=errs, mean_px=mean, max_px=worst,
                                         max_diff_vs_step2_px=max_diff, depth_camera_check=depth_check, passed=ok))
    print(f"仿真预测角点 vs 真机检测角点：每张 RMS 平均 {mean:.2f} px，最差 {worst:.2f} px（{len(errs)} 张；含机械臂绝对误差）")
    print(f"与 step2（OpenCV）同一指标的最大差异：{max_diff:.3f} px（要求 < 0.5 px）")
    if depth_check:
        print(f"仿真深度相机位姿 vs 彩色·出厂外参：{depth_check['trans_err_m'] * 1000:.3f} mm / {depth_check['rot_err_deg']:.4f}°")
    print("结论：" + ("✅ 标定结果在仿真里正确" if ok else "❌ 仿真与真机不一致，检查 step2 结果和配置"))
    print(f"对照图：{out}")
    return dict(mean_px=mean, max_px=worst, max_diff_vs_step2_px=max_diff, passed=ok, depth_check=depth_check)


if __name__ == "__main__":
    main()
