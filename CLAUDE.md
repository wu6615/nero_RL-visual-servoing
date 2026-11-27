# NERO7 眼在手上 RL 视觉伺服（my_RL）

NERO7 + 腕部 DC1，GAP-RL 思路的 RL 动态抓取。MuJoCo 仿真 + SB3 SAC。目录照 GAP-RL 组织（见 README.md）。

## 约定

- 与用户用中文交流；文档、终端提示、报错用中文。模块首行英文 docstring；关键逻辑用 `# 【说明】` 中文块注释（GAP-RL 风格）。
- 依赖以 `.venv` 和 `requirements.txt` 为准；不要改 NERO 工程（/home/yunfeiwu/adddisk1/nero_conveyor_ros2-main）或其他项目的环境，只读引用。例外（2026-11-26 用户要求）：给本项目采集示范数据的腕部 DC1 同步录制放在 NERO 工程（`src/nero_core/nero_core/wrist_dc1/`、`tools/run_hardware_continuous.sh --record-wrist-dc1`），改它时遵守 NERO 自己的 CLAUDE.md；转换成 RL 数据的代码留在本项目 `sim2real/demos/`。
- 新增或修改逻辑配 `tests/` 离线测试（不连相机、CAN）。改了环境要跑脚本基线评估确认仍可解。
- 现场几何值的出处记在 `docs/现场几何来源.md`；不要为了让测试通过修改实测值。

## 常用命令

```bash
source .venv/bin/activate
python -m pytest
python -m nero_vs_rl.algorithms.scripts.sac_eval --episodes 30            # 脚本基线（默认配置 = dc1_site.yaml 实测值；默认打开 MuJoCo 图形界面，--no-viewer 关闭）
python -m nero_vs_rl.algorithms.scripts.sac_train --run-name <名字>         # 训练
python -m nero_vs_rl.sim2real.deploy.step4_closed_loop_sim --episodes 10    # 真机闭环代码 + MuJoCo 虚拟实验室干跑（改了 deploy/runtime 后跑）
```

## 输出位置

训练日志、模型、评估视频放数据盘 `/home/yunfeiwu/adddisk1/MY_RL_test/simulation`（`nero_vs_rl.SIM_OUTPUT_DIR`，`train.log_dir` 默认值）；系统盘空间不够，不要往工程目录的 `runs/` 写大文件。真机录制（部署离线阶段）放 `/home/yunfeiwu/adddisk1/MY_RL_test/real`（`nero_vs_rl.REAL_DATA_DIR`）。

## 现场标定

步骤和验收标准在 `nero_vs_rl/sim2real/calibration/README.md`；结果只通过 step 脚本的 `--apply` 写入 `algorithms/config/dc1_site.yaml`，不要手改其中数值。训练/评估默认读它（`nero_vs_rl.TRAIN_CONFIG`）；`default.yaml` 是占位参数，只作对照。干跑（`--sim`）不要加 `--apply`。

## 真机安全

会让机械臂运动的程序只在用户明确要求时运行；不自动使能机械臂，不修改 CAN 状态（`ip link set can0 ...`）。
