"""One NERO run's data (wrist DC1 recording + cycle logs) -> RL demonstrations (observation dict, action) per grasp.

  python -m nero_vs_rl.sim2real.demos.build_demos --run /home/yunfeiwu/adddisk1/MY_RL_test/real/nero_demos/continuous_20261126_150002_12345

数据由 NERO 工程采集（实验台勾“录腕部 DC1”，或 tools/run_hardware_continuous.sh --record-wrist-dc1），
直接录到 REAL_DATA_DIR/nero_demos/<运行目录名>/；本脚本把 demos.npz、demos_summary.json 写在同一个目录里。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ... import REAL_DATA_DIR, TRAIN_CONFIG
from ...envs.observation import assemble_obs, gripper_points
from ...utils.config import load_config
from ..deploy.common import JointSource, RobotTwin, load_record, read_depth, write_json
from ..deploy.runtime.devices import width_to_q
from ..deploy.runtime.episode import LivePerception
from .actions import joints_to_action
from .nero_log import CommandTimeline, load_belt_speed, load_run

# =============================================================================
# 【说明】一件工件 → 一个示范回合（和真机闭环 runtime/episode.py 的一拍做同样的事，只是动作来自 NERO）：
#
#   窗口  NERO 接近段（intercept）开始前 pre_s 秒 ～ 接近段结束（闭爪后的跟随减速也在里面）
#   起点  窗口里第一次"完整检出"物体的那一帧（同部署的 WAIT → SERVO 判据；在这之前策略不会动）
#   每一拍（录制是 10 Hz，一帧一拍）：
#     观测  感知用 LivePerception（帧曝光时刻的关节 → 相机位姿 → 检出 → 记忆/测速），
#           TCP 位姿用当前实测关节，夹爪开口用实测宽度，上一步动作用上一拍的示范动作 —— 和部署时拼法完全一样
#     动作  NERO 在这一刻和一个控制周期之后被要求到的关节 → RL 动作（actions.joints_to_action）；
#           夹爪取一个周期之后生效的 NERO 夹爪指令。机械臂静止、NERO 没在发指令时，用实测关节当指令（动作 ≈ 0）
#   终点  窗口结束（不含之后的原地保持、搬运、返回：那些不是 RL 的任务；抬起 6 cm 由 RL 自己在仿真里学）
#   成功  NERO 自己按开口和力确认夹住（events.json 的 grip_confirmed_from_width_force）
#
# 时钟：录制和 NERO 日志都是本机 time.monotonic，直接对齐。程序会检查实测关节是否跟着 NERO 的指令走
#   （最佳滞后应在 0.05～0.2 s），对不上说明两份数据不是同一次运行。
# 带速：每个回合挂上 NERO 冻结这一件时用的 Uno 带速和基座系速度向量（belt_speed.json，按目标身份对应）。
#   策略观测里没有带速（RL 只靠视觉测速），它是标签：按带速筛数据、核对 RL 自己的视觉测速准不准
#   （每拍另存了 tracker 的估计速度 obj_vel_est，汇总里给出两者对比）。
# 奖励不在这里算：仿真奖励要用真值抓取，真机没有；每拍另存了物体中心、TCP、夹爪开口，训练时需要再算。
# =============================================================================
OBS_KEYS = ("tcp_pose", "gripper_pos", "action", "grasp_vis", "origin_gripper_pts", "gripper_pts_diff", "grasp_scores")
LAG_SEARCH_S = np.arange(0.0, 0.31, 0.01)


# 【说明】输入目录的两种结构：
#   新（2026-11-26 起，默认）  REAL_DATA_DIR/nero_demos/<运行目录名>/：录制在目录本身，执行日志在 nero_cycles/
#                              （录制结束时 NERO 侧复制过来；没复制成功时按 meta.json 的 nero_run_dir 去 NERO 工程里读）
#   旧                        NERO 运行目录本身：录制在 wrist_dc1/，执行日志就是 cycle_*/；输出另写到 REAL_DATA_DIR/nero_demos/
def resolve_inputs(run: Path) -> tuple[Path, Path, Path]:
    """(recording dir, directory holding cycle_*/, output dir) for one run."""
    if (run / "frames.jsonl").exists():
        cycles = run / "nero_cycles"
        if not any(cycles.glob("cycle_*")):
            meta, _ = load_record(run)
            cycles = Path(meta.get("nero_run_dir", ""))
            print(f"⚠ {run} 里没有复制好的执行日志，改读 NERO 工程里的 {cycles}")
        return run, cycles, run
    if (run / "wrist_dc1" / "frames.jsonl").exists():
        return run / "wrist_dc1", run, REAL_DATA_DIR / "nero_demos" / run.name
    raise SystemExit(f"{run} 里没有腕部录制：NERO 那次运行要勾“录腕部 DC1”（或加 --record-wrist-dc1）")


class RecordedArm:
    """The recording's 100 Hz joint history behind the ``joints_at`` call LivePerception makes on a live arm."""

    def __init__(self, joints: JointSource):
        self.joints_at = joints.at


def command_lag(joints: JointSource, timeline: CommandTimeline) -> tuple[float, float]:
    """(best lag s, rms rad): how far the measured joints trail NERO's commands; checks the two logs line up."""
    t = timeline.t[(timeline.t > joints.t[0] + 0.5) & (timeline.t < joints.t[-1] - 0.5)][::5]
    if len(t) < 20:
        raise SystemExit("录制和 NERO 运行日志在时间上没有重叠：是不是同一次运行？（两边都应是本机 time.monotonic）")
    meas = np.array([joints.at(x) for x in t])
    rms = [np.sqrt(np.mean([(m - timeline.q_at(x - lag)) ** 2 for m, x in zip(meas, t)
                            if timeline.q_at(x - lag) is not None])) for lag in LAG_SEARCH_S]
    i = int(np.nanargmin(rms))
    return float(LAG_SEARCH_S[i]), float(rms[i])


def build_episode(cycle, frames, record, cfg, twin, perception, joints, timeline, pre_s, rng, speeds) -> dict | None:
    """One NERO cycle -> list of ticks (obs, action, info), or None when the window has no usable frames."""
    st = cycle.intercept
    if st is None or cycle.t_close is None:
        return None
    win = [r for r in frames if st.t_start - pre_s <= r["stamp"] <= st.t_end]
    if not win:
        return None
    arm, dt, fk = RecordedArm(joints), twin.control_dt, twin.controller.ik.fk
    perception.reset()
    prev_action, ticks, started = np.zeros(7, dtype=np.float32), [], False
    for rec in win:
        stamp = rec["stamp"]
        frame = SimpleNamespace(stamp=stamp, depth=read_depth(record, rec))
        seen = perception.update(frame, arm, stamp - win[0]["stamp"])
        started |= seen.usable
        if not started:                                           # 部署时策略也要等第一次完整检出
            continue
        q_meas = joints.at(stamp)
        width = float(rec.get("gripper_width", timeline.width_at(stamp)))
        twin.set_joints(q_meas, width_to_q(width))
        p, R = twin.tcp_pose()
        pts = gripper_points(cfg.grasp_obs, twin.K, rng, twin.fixed_pts)
        obs, _, _ = assemble_obs(p, R, seen.grasps, seen.scores, seen.visible, perception.tracker.age_norm,
                                 width_to_q(width) / cfg.gripper_max_q, prev_action, pts, twin.N)
        q_now = timeline.q_at(stamp)
        q_now = q_meas if q_now is None else q_now
        q_next = timeline.q_at(stamp + dt)
        q_next = q_now if q_next is None else q_next
        act = joints_to_action(fk, q_now, q_next, timeline.width_at(stamp + dt), cfg)
        center = seen.det.center if seen.det is not None else np.full(3, np.nan)
        ticks.append(dict(obs=obs, action=act.action, raw=act.raw, clipped=act.clipped, t=stamp - st.t_start,
                          visible=seen.visible, width=width, tcp=p, center=center,
                          vel_est=perception.tracker.est_obj_vel.copy()))
        prev_action = act.action
    if not ticks:
        return dict(cycle=cycle.index, ticks=[], why="窗口里一直没有完整检出物体（策略不会启动）")
    belt = speeds.get(cycle.target, {})
    vel = belt.get("velocity_base_m_s")
    return dict(cycle=cycle.index, ticks=ticks, success=cycle.grip_confirmed, t_close=cycle.t_close - st.t_start,
                belt_speed=float(belt.get("speed_m_s") or np.nan),
                belt_velocity=np.asarray(vel, float) if vel else np.full(3, np.nan), why="")


def vision_speed(ep: dict) -> float:
    """Speed of RL's visual estimate at the last frame the object was seen: the one memory extrapolation then uses.

    不取所有可见帧的中位数：测速要攒够 3 帧才有结果，前两帧是 0；真机上 NERO 下降很快，
    一个回合常常只有 3～4 帧看得到，中位数会被这两个 0 拉低。不到 3 帧完整画面就测不出速度，给 NaN，
    不当成 0（否则汇总会显示“比 Uno 偏 −100%”，像是测错了，其实是没测）。
    """
    v = [np.linalg.norm(t["vel_est"]) for t in ep["ticks"] if t["visible"]]
    return float(v[-1]) if len(v) >= 3 else float("nan")


def save(out: Path, episodes: list[dict]):
    ticks = [(k, t) for k, ep in enumerate(episodes) for t in ep["ticks"]]
    arrays = {f"obs_{key}": np.stack([t["obs"][key] for _, t in ticks]) for key in OBS_KEYS}
    arrays.update(action=np.stack([t["action"] for _, t in ticks]), action_raw=np.stack([t["raw"] for _, t in ticks]),
                  clipped=np.array([t["clipped"] for _, t in ticks]), episode=np.array([k for k, _ in ticks]),
                  t=np.array([t["t"] for _, t in ticks]), visible=np.array([t["visible"] for _, t in ticks]),
                  gripper_width=np.array([t["width"] for _, t in ticks]), tcp=np.stack([t["tcp"] for _, t in ticks]),
                  obj_center=np.stack([t["center"] for _, t in ticks]), obj_vel_est=np.stack([t["vel_est"] for _, t in ticks]),
                  episode_last=np.array([i == len(ticks) - 1 or ticks[i + 1][0] != k for i, (k, _) in enumerate(ticks)]),
                  episode_cycle=np.array([ep["cycle"] for ep in episodes]),
                  episode_success=np.array([ep["success"] for ep in episodes]),
                  episode_belt_speed=np.array([ep["belt_speed"] for ep in episodes]),
                  episode_belt_velocity=np.stack([ep["belt_velocity"] for ep in episodes]))
    np.savez_compressed(out, **arrays)


def main(argv=None):
    ap = argparse.ArgumentParser(description="一次 NERO 运行的数据（腕部录制 + 执行日志）→ RL 示范（观测、动作）")
    ap.add_argument("--run", "--nero-run", dest="run", required=True,
                    help="REAL_DATA_DIR/nero_demos/<运行目录名>（或 NERO 工程的运行目录，旧结构）")
    ap.add_argument("--out", default=None, help="输出目录（默认就是数据目录本身；旧结构写到 REAL_DATA_DIR/nero_demos/）")
    ap.add_argument("--config", default=str(TRAIN_CONFIG), help="和要训练的策略用同一份配置（动作缩放、观测尺寸）")
    ap.add_argument("--pre-s", type=float, default=2.0, help="接近段开始前多少秒开始找物体")
    ap.add_argument("--latency", type=float, default=None, help="相机延迟（默认用配置里的 camera.latency_s）")
    args = ap.parse_args(argv)
    record, cycles_dir, out = resolve_inputs(Path(args.run))
    out = Path(args.out) if args.out else out
    out.mkdir(parents=True, exist_ok=True)
    cfg, _ = load_config(args.config)
    meta, frames = load_record(record)
    if frames and "gripper_width" not in frames[0]:
        print("⚠ 录制里没有夹爪开口：观测里的夹爪开口改用 NERO 的夹爪指令代替")
    cycles = load_run(cycles_dir)
    speeds = load_belt_speed(record / "belt_speed.json")
    if not speeds:
        print("⚠ 没有 belt_speed.json（NERO 侧没整理带速）：示范回合的带速记为 NaN")
    timeline, joints = CommandTimeline(cycles), JointSource(record, frames)
    lag, rms = command_lag(joints, timeline)
    print(f"NERO 周期 {len(cycles)} 个；实测关节落后指令 {lag * 1000:.0f} ms（残差 {rms:.4f} rad）")
    twin = RobotTwin(cfg)
    di = meta["camera"]["depth_intrinsics"]
    K = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
    D = np.asarray(di.get("dist") or [0.0] * 5, float)
    latency = cfg.camera.latency_s if args.latency is None else args.latency
    perception = LivePerception(cfg, twin, K, D, latency)
    rng = np.random.default_rng(0)
    episodes, skipped = [], []
    for c in cycles:
        ep = build_episode(c, frames, record, cfg, twin, perception, joints, timeline, args.pre_s, rng, speeds)
        if ep is None:
            skipped.append(dict(cycle=c.index, why="录制没覆盖这一件，或 NERO 这一件没有接近段 / 没闭爪"))
        elif not ep["ticks"]:
            skipped.append(dict(cycle=c.index, why=ep["why"]))
        else:
            episodes.append(ep)
    twin.close()
    if not episodes:
        # 不留上一次（可能是旧感知规则）生成的 demos.npz，免得被当成这次的结果
        (out / "demos.npz").unlink(missing_ok=True)
        summary = dict(record=str(record), cycle_logs=str(cycles_dir), config=args.config, episodes=0, ticks=0,
                       skipped=skipped, out=str(out))
        write_json(out / "demos_summary.json", summary)
        print("没有生成任何示范回合：" + "；".join(f"周期 {s['cycle']}：{s['why']}" for s in skipped[:5]))
        return summary
    save(out / "demos.npz", episodes)
    n_ticks = sum(len(ep["ticks"]) for ep in episodes)
    clipped = sum(t["clipped"] for ep in episodes for t in ep["ticks"])
    summary = dict(record=str(record), cycle_logs=str(cycles_dir), config=args.config, camera_latency_s=latency,
                   command_lag_s=lag, command_lag_rms_rad=rms, episodes=len(episodes), ticks=n_ticks,
                   success=int(sum(ep["success"] for ep in episodes)), clipped_ticks=int(clipped),
                   visible_frac=float(np.mean([t["visible"] for ep in episodes for t in ep["ticks"]])),
                   per_episode=[dict(cycle=ep["cycle"], ticks=len(ep["ticks"]), success=ep["success"],
                                     t_close=round(ep["t_close"], 3), clipped=int(sum(t["clipped"] for t in ep["ticks"])),
                                     t_first_seen=round(ep["ticks"][0]["t"], 3),
                                     visible_ticks=int(sum(t["visible"] for t in ep["ticks"])),
                                     belt_speed_cm_s=round(ep["belt_speed"] * 100, 2),
                                     vision_speed_cm_s=round(vision_speed(ep) * 100, 2)) for ep in episodes],
                   skipped=skipped)
    write_json(out / "demos_summary.json", summary)
    print(f"示范回合 {len(episodes)} 个（跳过 {len(skipped)}），共 {n_ticks} 拍；NERO 确认夹住 {summary['success']} 个；"
          f"看得到物体的拍 {summary['visible_frac']:.0%}；动作超出 RL 范围（被截断）的拍 {clipped}（{clipped / n_ticks:.0%}）")
    pairs = [(e["belt_speed_cm_s"], e["vision_speed_cm_s"]) for e in summary["per_episode"]
             if np.isfinite(e["belt_speed_cm_s"]) and np.isfinite(e["vision_speed_cm_s"])]
    if pairs:
        err = np.array([v - u for u, v in pairs])
        print(f"带速：Uno {np.median([u for u, _ in pairs]):.1f} cm/s；RL 视觉测速（最后一个可见帧）比 Uno 偏 "
              f"{np.median(err):+.1f} cm/s（中位，{len(pairs)} 个回合）")
    else:
        print("带速：没有一个回合有 ≥ 3 帧完整画面，RL 视觉测速测不出来，无法和 Uno 比")
    seen =[e["t_first_seen"] for e in summary["per_episode"]]
    print(f"第一次完整看到物体：接近段开始后 {np.median(seen):+.2f} s（中位）；每回合看得到的帧中位 "
          f"{np.median([e['visible_ticks'] for e in summary['per_episode']]):.0f} 帧")
    print(f"输出：{out / 'demos.npz'}、{out / 'demos_summary.json'}")
    summary["out"] = str(out)
    return summary


if __name__ == "__main__":
    main()
