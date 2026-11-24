"""Closed-loop execution layer: the online counterpart of step2_replay, shared by the lab and the MuJoCo dry run.

  devices.py   接口：机械臂 / 相机 / 时钟要提供什么（真机 hardware/nero_control.py，仿真 hardware/sim_workcell.py）
  streamer.py  100 Hz 关节插值发送
  safety.py    安全档位 + 每个目标发出前的检查
  episode.py   一个回合：等目标 → 策略伺服 → 抓起后停住（每拍：感知 → 观测 → 策略 → 控制 → 安全 → 发送）
  session.py   一次运行：孪生 + 安全 + 发送只建一次，跑多个回合，回合之间回起点
  run_log.py   日志（逐拍 + step1 录制格式的深度帧，可用 step2 回放）
"""
