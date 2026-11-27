# sim2real

对应 GAP-RL `sim2real/`。

- **已就绪**：设备驱动 `hardware/`（DC1、NERO 只读反馈、仿真替身）和现场标定 `calibration/`——**操作步骤见 [calibration/README.md](calibration/README.md)**。
- **离线阶段已写**：`deploy/`（录制 + 离线回放：感知 → 观测 → 策略，机械臂不动）——**操作步骤见 [deploy/README.md](deploy/README.md)**；2026-11-22 第一次真机录制：尺寸误差 ≤ 2.3 mm，相机没被碰动，策略看得到物体时 100% 朝向抓取（详见 deploy/README.md）。
- **闭环阶段已写、仿真干跑通过、真机未运行**：执行层 `deploy/runtime/`（100 Hz 插值发送、安全层、回合状态机）+ 唯一会发运动指令的 `hardware/nero_control.py` + 仿真虚拟实验室 `hardware/sim_workcell.py`——**操作步骤见 [deploy/闭环阶段.md](deploy/闭环阶段.md)**（step4 仿真干跑 → step5 第一次运动 → step6 跟随不抓 → step7 抓取）。
- **示范数据（2026-11-26 写好，真机未录）**：NERO 实验台启动抓取时勾“录腕部 DC1”，腕部 DC1 + 关节 + 夹爪 + 执行日志直接录到 `MY_RL_test/real/nero_demos/`（采集在 NERO 工程），`demos/` 把它转成 RL 的观测和动作——**操作步骤见 [demos/README.md](demos/README.md)**。

## 链路

```text
DC1 RGB-D ──→ 点云（深度 → 相机系 → TCP 系，手眼标定 T_tcp_cam）
          ──→ 目标分割 + 抓取检测（替换 grasp_detection 的解析抓取）
          ──→ 抓取跟踪 / 记忆外推（与 envs/dynamic_grasp.py::_observe_grasps 同逻辑）
          ──→ 组装与仿真完全相同的观测字典
          ──→ SAC 策略（10 Hz） ──→ EEDeltaPoseController（同一份 IK）
          ──→ 安全修正与检查（指尖离带面、软限位、工作空间、孪生模型碰撞）
          ──→ 关节目标 ──→ 100 Hz 插值发送（deploy/runtime/streamer.py）──→ move_js（hardware/nero_control.py）
```

## 待办

1. ~~DC1 驱动~~：`hardware/dc1_camera.py` + `hardware/orbbec_v1.py`（Orbbec SDK v1.10.37，ctypes；深度和彩色同一帧组）。2026-11-21 真机只读验证：SN CC1N16201EN，640x400 深度 + 1280x720 彩色，出厂标定记录匹配，约 24 帧/秒（USB2.0）。SDK v2 枚举不到 DC1，已删除。
2. ~~眼在手上手眼标定、盲区、带位置~~：`calibration/` step0～step5，结果写入 `algorithms/config/dc1_site.yaml`。仿真干跑已验证（与真值差 0.02 mm），**真机未验证**。
3. ~~基座系核对~~：step0 / step2 比较 SDK 法兰位姿与 MJCF 正解。
4. ~~执行层~~：`deploy/runtime/`。100 Hz 本机插值后用 `move_js` 发送（每点 ≤ 档位限速 × 10 ms；NERO 工程现场同样是 js + 100 Hz），碰撞检查在孪生模型上逐目标同步做。**真机未运行**。
5. 仿真里加 DC1 的碰撞外形（现在相机只是一个视点，没有体积）。
6. ~~先离线~~：`deploy/step1_record` + `step2_replay`（2026-11-22 写好，仿真干跑：位置误差 0.9 mm、尺寸 0.1 mm）。真机录制待做；之后低速空跑。
7. ~~闭环前处理两个已知差异~~：回合状态机在第一次完整检出物体后才启动策略（`runtime/episode.py` 的 WAIT 阶段）。

## 安全

会让机械臂运动的程序只能在操作者明确要求时运行；不自动使能机械臂，不修改 CAN 状态。
