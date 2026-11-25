# Isaac Sim 训练框架

在云端 GPU 服务器上用 Isaac Lab 批量训练；策略接口和 MuJoCo 版完全一样，训好的策略搬回本机，用现有的 step4 干跑和真机流程验证。

> **状态（2026-11-25）：框架已搭好，Isaac 侧一行都没有运行过**（本机装不下 Isaac Sim）。
> 已在本机验证的是 `core/`（纯 PyTorch）：和 MuJoCo / numpy 版逐项对数，见 `tests/test_isaac_core.py`。
> `sim/` 和 `scripts/step1～step5` 对着 Isaac Lab 2.3.2 源码写，接口名核对过，第一次上服务器要按 step0 → step4 逐步验收。

## 1. 结构

```text
isaac/
  isaac.yaml            Isaac 专用参数：环境数、物理步长、关节驱动、SAC 超参
  export_assets.py      本机运行一次：从 MuJoCo 模型导出 assets/（机器人模型改了才重跑）
  assets/               nero7_kinematics.npz（运动学表、夹爪凸包）、nero7_isaac.urdf（网格引用 assets/descriptions/nero7/meshes）
  core/                 任务本体，纯 PyTorch，不 import Isaac 也不 import MuJoCo
    kinematics.py         关节角 → TCP、相机、雅可比、夹爪最低点
    controller.py         动作 → IK → 修正 → 关节目标
    objects.py            物体规格库（抓取直接用 analytic_grasps.py）
    trajectory.py         带上的运动（含静止、慢带）
    perception.py         可见性、噪声、抓取记忆
    observation.py        观测字典（键和形状同 MuJoCo 版）
    reward.py             奖励（v8：真正夹住才算、闭合姿态扣分）
    task.py               把上面几块串成一个任务；仿真器只需提供状态、执行目标
  sim/                  唯一依赖 Isaac Lab 的地方：scene.py（场景）、grasp_env.py（DirectRLEnv 接线）
  sb3_vec_env.py        给 SB3 用的批量环境（带真正的终止观测）
  policy_io.py          策略按张量搬运（避开 Python / SB3 版本差异）
  scripts/              step0～step7
```

任务参数仍在 `algorithms/config/default.yaml` + `dc1_site.yaml`，两边仿真共用一份；`isaac.yaml` 只放 Isaac 专用的。

## 2. 选型

| 项 | 选择 | 原因 |
|---|---|---|
| Isaac Lab / Isaac Sim | 2.3.2 / 5.1.0 | 2026-11 已有 3.0（Isaac Sim 6.2），但四元数改成 xyzw、数据接口也变了；2.3.2 用 PhysX，资料多。要升级只改 `sim/` 两个文件 |
| Python | 3.11 | Isaac Sim 5.1 要求 |
| GPU | 带 RT Core、显存 ≥ 16 GB（L40S、RTX 4090、RTX 6000 Ada 等） | A100 / H100 没有 RT Core |
| 内存、硬盘 | ≥ 32 GB，≥ 50 GB | 经验池约 2.5 GB |
| 训练 | SB3 SAC，同一个抓取点编码网络 | 策略能直接用回部署 |

## 3. 上传

本机打包（不含虚拟环境、标定图片、SDK；约十几 MB）：

```bash
cd /home/yunfeiwu/桌面 && tar czf /home/yunfeiwu/adddisk1/MY_RL_test/nero_isaac_upload.tar.gz --exclude=.venv --exclude=.git --exclude=vendor --exclude=calib_data --exclude=runs --exclude=Log --exclude=__pycache__ my_RL
```

上传到服务器的 home 目录并解压：

```bash
tar xzf ~/nero_isaac_upload.tar.gz -C ~
```

## 4. 服务器环境

```bash
conda create -y -n nero_isaac python=3.11 && conda activate nero_isaac
```

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab && cd ~/IsaacLab && ./isaaclab.sh --install sb3
```

输出默认在 `~/nero_runs`，用环境变量 `NERO_ISAAC_OUT` 改。

## 5. 步骤和验收

都在 `~/my_RL` 下运行。每一步通过了再走下一步。

| 步骤 | 验收 |
|---|---|
| step0 查机器 | 全部 ✓ |
| step1 URDF → USD | 打印生成的文件 |
| step2 核对资产 | body 位姿差 < 0.5 mm / 0.1°；物体被带面带走；两指接触能报出 |
| step3 关节驱动 | 滞后 80～120 ms（真机 90～100）；夹爪 0.2 s 合到 1.5 cm 以内 |
| step4 脚本基线 | 现场带速成功率 > 30% |
| step5 训练 | 先冒烟 20 万步能跑、成功率曲线在动，再正式训 |
| step6 / step7 | 导出、搬回本机 |
| **最终** | 本机 step4 闭环干跑（部署那套代码）；v5 也按 v8 的新标准重评后再比 |

```bash
cd ~/my_RL && python -m nero_vs_rl.isaac.scripts.step0_check_server
```

```bash
python -m nero_vs_rl.isaac.scripts.step1_convert_robot --headless
```

```bash
python -m nero_vs_rl.isaac.scripts.step2_check_assets --headless --num-envs 16
```

```bash
python -m nero_vs_rl.isaac.scripts.step3_check_drives --headless --num-envs 16
```

```bash
python -m nero_vs_rl.isaac.scripts.step4_scripted_baseline --headless --num-envs 128
```

冒烟训练：

```bash
python -m nero_vs_rl.isaac.scripts.step5_train --headless --run-name smoke --total-timesteps 200000
```

正式训练（Ctrl+C 暂停会连经验池一起存，再加 `--resume` 接着训）：

```bash
python -m nero_vs_rl.isaac.scripts.step5_train --headless --run-name isaac_v1
```

```bash
python -m nero_vs_rl.isaac.scripts.step6_export_policy --run-name isaac_v1
```

把 `~/nero_runs/isaac_v1/policy_best.pt` 拷到本机 `/home/yunfeiwu/adddisk1/MY_RL_test/simulation/isaac_v1/`，然后在本机：

```bash
cd /home/yunfeiwu/桌面/my_RL && source .venv/bin/activate && python -m nero_vs_rl.isaac.scripts.step7_import_policy --pt /home/yunfeiwu/adddisk1/MY_RL_test/simulation/isaac_v1/policy_best.pt
```

## 6. 和 MuJoCo 版有意不同的地方

| 项 | MuJoCo 版 | Isaac 版 |
|---|---|---|
| 物体尺寸、质量、摩擦 | 每回合随机 | 按环境固定（128 种规格）：PhysX 不能每回合改尺寸 |
| 物体与带面的摩擦 | 带面 1.1 | 物体规格里的 0.4～0.9（接触取较小值） |
| 手指遮挡 | 射线打碰撞网格（2026-11-25 修好，之前一直没生效） | 两指凸包求交，规则相同；起始状态看到与否 149/150 一致 |
| 夹爪两指 | 等式约束严格对称 | 两个独立驱动发对称目标，推偏时可能不对称 |
| 手指长度随机 | ±5 mm | 没做 |
| 相机延迟 | 没有 | 没有（复现实验里影响不大，以后再加） |
| link7 质量 | 1e-6 kg | 抬到 10 g（太小 PhysX 不稳） |
| 物理步长 | 2 ms | 5 ms |

## 7. 最可能出问题的地方（第一次上服务器先看这里）

1. **USD 里的 body 路径。** 接触传感器按 `/World/envs/env_.*/Robot/link1` 这样的路径过滤；导入器如果多套了一层，step2 会在建传感器时报错，改 `sim/scene.py` 的 `ROBOT`。
2. **带面是否带得动物体。** 用"重刚体每步拉回原点 + 设速度"的办法，step2 有专门一项核对。
3. **PhysX 的关节摩擦单位。** Isaac Sim 5.x 里按力矩（N·m）理解；step3 滞后明显偏大时，先把 `joint_friction` 调小。
4. **接触力读数。** 静态物体（护栏、底座）上的传感器是否上报运动学刚体的接触。step2 没有单独测这一项，训练日志里碰撞惩罚一直是 0 就要查。
