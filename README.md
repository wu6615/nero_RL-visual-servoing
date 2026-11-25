# NERO7 眼在手上 RL 视觉伺服动态抓取

AgileX NERO7 七轴机械臂 + 腕部松灵/奥比中光 DaBai DC1 双目相机（眼在手上），用强化学习做传送带上运动物体的视觉伺服抓取。算法思路来自 [GAP-RL](https://arxiv.org/abs/2410.03509)（Grasps As Points，RA-L 2024），目录结构也照 GAP-RL 组织；机器人模型、现场几何来自 `/home/yunfeiwu/adddisk1/nero_conveyor_ros2-main`。

当前阶段（2026-11-24）：

- **仿真**：v6 训练完成（对齐部署的任务上 86.7%，v5 同任务 74.7%）。v6 到真机闭环路径的干跑上不比 v5 好（现场带速 77.6% 对 89.5%），**真机先用 v5**，见下面"训练结果"。
- **真机**：现场标定、离线录制回放、第一次运动（step5）、策略跟随（step6）都已通过；第一次闭环抓取（step7，静止物体）抓起 1 回合，主要问题是 v5 没见过静止物体。记录见 [nero_vs_rl/sim2real/deploy/闭环阶段.md](nero_vs_rl/sim2real/deploy/闭环阶段.md)。
- **下一步**：仿真平台迁移到 Isaac Sim（租服务器）。要带走的参数、实测数据和建议见 [docs/IsaacSim迁移说明.md](docs/IsaacSim迁移说明.md)，数据位置见 [docs/数据清单.md](docs/数据清单.md)。

## 目录（与 GAP-RL 对照）

```text
nero_vs_rl/
  agents/robots/nero7.py            机器人句柄：关节/夹爪/TCP/接触         ← GAP-RL agents/robots
  agents/controllers/pd_ee_delta_pose.py  TCP 增量位姿 → DLS IK → 关节目标  ← agents/controllers
  algorithms/config/default.yaml    训练配置                               ← algorithms/config
  algorithms/Networks/              抓取点组编码器（GraspPointGroup）       ← algorithms/Networks
  algorithms/rl_utils.py            并行环境构造                           ← algorithms/rl_utils.py
  algorithms/scripts/sac_train.py   训练入口                               ← scripts/sac_train.py
  algorithms/scripts/sac_eval.py    评估入口（SAC 或脚本基线）              ← scripts/sac_*_eval.py
  algorithms/scripted_pbvs.py       脚本 PBVS 基线                         ← sim2real test_heuristic_plan
  assets/descriptions/nero7/        MJCF 场景 + NERO7 网格                  ← assets/descriptions
  envs/base_env.py                  MuJoCo 基类：加载、插值步进、渲染        ← envs/base_env.py
  envs/dynamic_grasp.py             动态抓取任务：物体、感知、奖励           ← envs/pick_single.py
  envs/observation.py               抓取记忆/测速外推/观测拼装（仿真真机共用）
  grasp_detection/                  抓取检测（现为解析抓取，LoG 的位置）     ← localgrasp/
  sensors/camera.py                 腕部 DC1：投影、可见性、盲区、RGB-D     ← sensors/
  sensors/pointcloud.py             深度 → 基座系点云、带面拟合
  sim2real/hardware/                DC1 驱动、NERO 只读反馈、NERO 运动控制（唯一会发运动指令）、仿真替身 ← robot_control.py / gripper_control.py
  sim2real/calibration/             现场标定 step0～step5（手眼、盲区、带位置）← sim2real/config.py 的手眼常数
  sim2real/deploy/                  部署：step1～3 离线（录制、回放、测延迟），step4～7 闭环（仿真干跑、第一次运动、跟随、抓取）
  sim2real/deploy/runtime/          闭环执行层：100 Hz 发送、安全层、回合状态机、日志 ← sim2real_traj_following.py 主循环
  utils/                            配置 dataclass、SE(3)、轨迹             ← utils/
  examples/                         轨迹可视化、腕部相机画面                 ← examples/
tests/                              离线测试（不连相机/CAN）
docs/                               设计说明、现场几何来源
```

GAP-RL 的 `vector/`（SAPIEN 并行封装）不需要：MuJoCo 环境很轻，直接用 SB3 `SubprocVecEnv`。`data/`（预采抓取数据集）暂不需要：解析抓取每步现算只要微秒级。

## 环境

工程自带 `.venv`（Python 3.10，torch 2.14.1+cu130，MuJoCo 3.17，gymnasium 1.2，SB3 2.9），不影响其他项目：

```bash
cd /home/yunfeiwu/桌面/my_RL
source .venv/bin/activate
python -m pytest                    # 84 个离线测试，约 5 分钟（不连相机、CAN）
```

从零重建：`python3 -m venv .venv && .venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`；真机/标定另装 `requirements-hardware.txt`（其中有 pyAgxArm 的安装说明）。

## 常用命令

```bash
# 看环境：脚本 PBVS 抓取的 MuJoCo 窗口 / 视频
python -m nero_vs_rl.examples.test_env_trajs --mode line --viewer
python -m nero_vs_rl.examples.test_env_trajs --mode bezier2d --video /home/yunfeiwu/adddisk1/MY_RL_test/simulation/demo.mp4
# 腕部相机 RGB + 深度（带盲区）PNG
python -m nero_vs_rl.examples.view_wrist_camera --out /home/yunfeiwu/adddisk1/MY_RL_test/simulation/wrist

# 以下命令默认用现场实测配置 dc1_site.yaml（启动时会打印"配置: ..."）；对照占位参数加 --config nero_vs_rl/algorithms/config/default.yaml
# 脚本基线评估（验证环境可解）
python -m nero_vs_rl.algorithms.scripts.sac_eval --episodes 30          # 默认打开 MuJoCo 图形界面实时播放（--playback 2 两倍速，--no-viewer 不开）

# 训练 SAC（默认 2M 步、8 个并行环境；输出在 /home/yunfeiwu/adddisk1/MY_RL_test/simulation/<名字>/，TensorBoard 在 tb/）
python -m nero_vs_rl.algorithms.scripts.sac_train --run-name sac_dc1_v2
# 训练中按 Ctrl+C = 暂停（存权重 + 经验池到 paused_model.zip / paused_replay_buffer.pkl），之后接着训：
python -m nero_vs_rl.algorithms.scripts.sac_train --run-name sac_dc1_v7 --resume
# 或者从已有模型的权重接着训（例如占位参数下训好的 sac_dc1_v1），比从零开始快
python -m nero_vs_rl.algorithms.scripts.sac_train --run-name sac_dc1_v2_ft --init-model /home/yunfeiwu/adddisk1/MY_RL_test/simulation/sac_dc1_v1/best/best_model.zip
tensorboard --logdir /home/yunfeiwu/adddisk1/MY_RL_test/simulation
# 评估训练好的模型
python -m nero_vs_rl.algorithms.scripts.sac_eval --model /home/yunfeiwu/adddisk1/MY_RL_test/simulation/sac_dc1_v5/best/best_model.zip --video /home/yunfeiwu/adddisk1/MY_RL_test/simulation/eval.mp4
```

## 云端 Isaac Sim 训练

框架在 [nero_vs_rl/isaac/](nero_vs_rl/isaac/README.md)（2026-11-25 搭好，Isaac 侧还没有运行过）。策略接口不变，训好后搬回本机用 step4 干跑和真机流程验证。

## 现场标定

DC1 装到夹爪上之后，按 [nero_vs_rl/sim2real/calibration/README.md](nero_vs_rl/sim2real/calibration/README.md) 的 step0～step5 操作（全程只读机械臂，由操作者摆姿态）。手眼标定、近距盲区、带的位置都会写进 `nero_vs_rl/algorithms/config/dc1_site.yaml`（`base: default.yaml` + 标定值），训练、评估和 `view_wrist_camera` 默认就读这个文件，仿真直接用上真机的相机位姿和带的位置。

## 真机测试

标定之后分两个阶段，所有命令都写在文档里，可以直接复制：

1. **离线阶段**（机械臂不动）：[nero_vs_rl/sim2real/deploy/README.md](nero_vs_rl/sim2real/deploy/README.md)。录 DC1 深度和关节角，在电脑上回放"感知 → 观测 → 策略"，检查真机上策略看到的东西和仿真一致。
2. **闭环阶段**（机械臂会动）：[nero_vs_rl/sim2real/deploy/闭环阶段.md](nero_vs_rl/sim2real/deploy/闭环阶段.md)。step4 仿真干跑 → step5 第一次运动 → step6 跟随不抓 → step7 抓取。

## 已验证（仿真，2026-11-21）

| 项 | 结果 |
|---|---|
| 离线测试 | 29/29 通过（含 8 个标定测试） |
| 标定干跑（仿真相机/机械臂） | 手眼结果与真值差 0.02 mm / 0.07°；放进仿真后角点预测误差 0.08 px；带位置测量误差 < 6 mm |
| 脚本 PBVS 基线，30 回合 | static 90%、line 100%、bezier2d 90%，平均首次成功 3.5–3.7 s |
| 仿真速度 | 约 2.5 ms/控制步（单进程，不渲染） |
| SAC 冒烟训练 | 4 并行环境 3000 步，GPU 上约 165 步/s，checkpoint / eval / best 模型都能保存 |

**未验证**（2026-11-21 时）：真机闭环抓取。完整训练的成功率见下一节；真机闭环结果见闭环阶段.md。DC1 经 Orbbec SDK v1 取流已在真机上验证（2026-11-21，只读）。

## 训练结果（仿真）

### v6：对齐部署的任务（2026-11-24）

v6 的任务在 v5 上改了这些，都是为了和真机闭环一致：
- 物体从带宽内任意横向位置、任意朝向过来（v5 只在中线 ±7 cm）；
- 顶面完整在画面里才算看到；
- 训练里也做部署时的目标修正：拉回工作空间、指尖抬高、软限位；
- 手指碰护栏扣分；
- 抓起后要停住；
- 起始位姿往远侧挪 8 cm、抬到 28 cm，相机能看到整条带。

细节见 [docs/设计说明.md](docs/设计说明.md)。

下表都是同一批 300 回合（`--seed 20000 --episodes 300 --no-viewer`），在 v6 任务上评估：

| 模型 | 总成功率 | <8 | 8～14 | 14～20 | 20～30 | 30～40 | ≥40 cm/s | 现场带速 ≤15 | 失败（漏抓 / 掉落 / 超时） |
|---|---|---|---|---|---|---|---|---|---|
| **sac_dc1_v6 best**（从头 200 万步，最好的就是最后一步） | **86.7%** | 88% | 96% | 87% | 91% | 82% | 78% | **93.6%** | 32 / 7 / 1 |
| sac_dc1_v5 best（v5 任务上训练） | 74.7% | | | | | | | 85.9% | 以漏抓为主 |
| 脚本 PBVS 基线 | 19% | | | | | | | 63% | |

训练中每 10 万步在固定的 100 回合上评估一次：
- 40 万步 40%；
- 100 万步 81%；
- 150 万步 87%；
- 200 万步 90%。

到最后还在涨，再训更久可能还有提升。

**真机闭环路径的干跑**：`step4_closed_loop_sim --profile grasp`，和真机同一套感知、安全修正与检查、100 Hz 插值。

第一轮（种子 20000 起 100 回合）里，v6 有 20 回合被"指令目标位姿下手指会碰护栏"的检查停下，成功率只有 63%（v5 71%）。

- 训练环境里同一批种子，v6 的手指实际碰护栏 0 次。
- 部署的检查看的是**指令目标**，它比实际机械臂领先约 0.2 s。v6 张着夹爪去抓靠边的物体时，外侧指尖的目标会伸进护栏。
- 现在部署时改成修正：夹爪在护栏正上方的部分抬到护栏顶 + 3 mm 以上，不再直接停（`deploy/runtime/safety.py` 的 `project`；训练环境不变）。

修正后，种子 20000～20299 共 300 回合：

| 模型 | 真值成功率 | 现场带速 ≤15 | 抓起 / 安全停止 / 超时 | 抓起判断误报 |
|---|---|---|---|---|
| **sac_dc1_v5 best** | **76.0%**（228/300） | **89.5%**（68/76） | 230 / 0 / 69（另 1 回合没检出） | 2 |
| sac_dc1_v6 best | 73.0%（219/300） | 77.6%（59/76） | 235 / 11 / 53（另 1 回合没检出） | 17 |

真机安全带面按 −0.112 算（配置 −0.12 和标定实测取更高的）。如果实际带面就在 −0.12，指尖会比训练多抬 8 mm，两个模型都会掉点。`--lab-belt` 就是按真机安全带面跑的，种子 20000 起 100 回合：

| 模型 | 真值成功率 | 现场带速 ≤15 |
|---|---|---|
| v5 | 62%（仿真带面时 72%） | 84.6%（88.5%） |
| v6 | 65%（70%） | 80.8%（76.9%） |

结论：
- **真机先用 v5。** v6 纯仿真高 12 个百分点，但到闭环路径上不比 v5 好，现场带速下还低 12 个百分点。
- v6 的"抓起判断误报"也多：夹爪闭合、抬高了，但其实没夹住。
- v6 从仿真 86.7% 掉到闭环 73%，v5 几乎不掉（74.7% → 76%），原因还没查清。

### v5（2026-11-23）

现场配置 dc1_site.yaml：带面会动（掉回带上的物体随带走），两侧 1.5 cm 护栏，带速 4～50 cm/s，8 s（80 步）限时，厂家关节限速。物体只在带中线 ±7 cm。下表都是同一批 300 回合（`--seed 20000 --episodes 300 --no-viewer`），按带速分档：

| 模型 | 总成功率 | <8 | 8～14 | 14～20 | 20～30 | 30～40 | ≥40 cm/s | 失败（漏抓 / 掉落 / 超时） |
|---|---|---|---|---|---|---|---|---|
| **sac_dc1_v5 best**（从头 200 万步，最佳在约 105 万步） | **88.3%** | 96% | 96% | 92% | 91% | 81% | 82% | 33 / 1 / 1 |
| sac_dc1_v5 final（200 万步） | 88.3% | 92% | 92% | 95% | 91% | 84% | 82% | 28 / 6 / 1 |
| sac_dc1_v5_ft best（v4 权重微调） | 81.0% | 96% | 88% | 89% | 82% | 69% | 76% | 52 / 5 / 0 |
| sac_dc1_v5_ft 70 万步 | 82.7% | 88% | 96% | 89% | 82% | 78% | 71% | 43 / 6 / 3 |
| sac_dc1_v4（旧环境训练，放到新环境） | 68.3% | | | | | | | 以漏抓为主 |

推荐用 **sac_dc1_v5/best/best_model.zip**（部署回放 step2 的默认模型）。抓到后中位第 9 步（0.9 s）首次成功；抬起后把物体移到工作空间靠机器人一侧的上角（x≈−0.18、z≈0.28 m，都是工作空间上限）停住，60 回合没有碰护栏。失败主要是高带速（≥30 cm/s）下漏抓。

## 占位值（装好 DC1 后要换）

| 项 | 现值 | 怎么得到真实值 |
|---|---|---|
| 腕部相机安装位姿 | robot.xml 里的仿真占位 | 标定 step1–2（手眼） |
| 内参 / 分辨率 | fovy 75°，640×400 | 标定 step2（DC1 深度出厂内参） |
| 近距盲区 | 0.15 m | 标定 step4 |
| 带中心线 x、带宽 | −0.40 m（等待位估计）、0.30 m | 标定 step5 |

实测/有据可查的值（带面高度、带向、等待位、跟随区间）的出处见 [docs/现场几何来源.md](docs/现场几何来源.md)。设计细节见 [docs/设计说明.md](docs/设计说明.md)。

## 数据下载

真机和仿真数据太大（约 1.3 GB），放在 GitHub Release [`data-20261125`](https://github.com/wu6615/nero_RL-visual-servoing/releases/tag/data-20261125)，不进 git：

| 附件 | 内容 |
|---|---|
| `real_data_20261125.tar.gz.part00`～`part05` | 真机：离线录制（2026-11-22）、step5 运动检查、step6 跟随、step7 抓取（2026-11-24）。深度图、彩色图、关节、逐拍日志 |
| `simulation_data_20261125.tar.gz.part00`～`part19` | 仿真：v0～v7 的模型、检查点、评估和 TensorBoard 日志，闭环干跑日志，真机回合复现结果 |
| `SHA256SUMS.txt` | 合并后两个压缩包的校验和 |

每个目录的说明见 [docs/数据清单.md](docs/数据清单.md)。附件按 40 MB 切块，下载后先合并：

```bash
gh release download data-20261125 --repo wu6615/nero_RL-visual-servoing
```

```bash
cat real_data_20261125.tar.gz.part* > real_data_20261125.tar.gz && cat simulation_data_20261125.tar.gz.part* > simulation_data_20261125.tar.gz && sha256sum -c SHA256SUMS.txt
```

```bash
tar xzf real_data_20261125.tar.gz && tar xzf simulation_data_20261125.tar.gz
```

标定原始图片（`calib_data/`，约 220 MB）没有上传，标定结果在 `nero_vs_rl/algorithms/config/dc1_site.yaml`。

## 来源和许可

NERO 网格与运动学来自 AgileX [agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf)（MIT，见 `assets/descriptions/nero7/LICENSE`、`SOURCE.md`），MJCF 适配沿用 nero_conveyor_ros2 工程（原工程无许可证）。网络结构与奖励设计参考 GAP-RL（RA-L 2024）。
