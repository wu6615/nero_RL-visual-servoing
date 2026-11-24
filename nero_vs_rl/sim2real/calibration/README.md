# 现场标定指南（DC1 眼在手上）

按编号顺序跑。所有程序都**只读**机械臂（不使能、不发运动指令、不改 CAN），机械臂由操作者用示教/拖动模式（或厂家上位机）摆姿态。每一步的数据存在 `calib_data/<时间>_<步骤>/`，结果自动写进 `nero_vs_rl/algorithms/config/dc1_site.yaml`，仿真和训练直接用它。

> **真机标定的位置能放进仿真吗？能。** 手眼标定解出的是"相机相对 `gripper_base` 的位姿"，而仿真里的 `wrist_camera` 正好挂在 `gripper_base` 上，所以结果原样写成 `env.camera.pos_in_gripper_base / quat_in_gripper_base`（OpenCV→MuJoCo 坐标轴转换由 `sim_bridge.py` 完成）。step3 会在训练用的同一个环境类里重新投影，验证仿真相机看到的和真机一致。

## 带去实验室

- 这台电脑（工程 + `.venv`）、DC1 和 USB3 线、USB-CAN
- 标定板：NERO 工程那块玻璃棋盘格，**12×9 个方格 = 11×8 个内角点，方格 20 mm**（其他板用 `--cols/--rows/--square` 改）
- 一块哑光平板（测盲区用，纸板即可）；胶带（固定标定板）

## 0. 准备（约 10 分钟）

1. DC1 刚性装到夹爪上，**标定之后不能再动**；动了就要重标。
2. DC1 只能用 **Orbbec SDK v1**（OpenNI 协议旧相机；SDK v2 / pyorbbecsdk2 枚举不到它，2026-11-21 实测）。
   SDK 放在 `vendor/orbbec_sdk_v1/OrbbecSDK_v1.10.37/`（不进 git）。**本机已经有了**（`ls vendor/orbbec_sdk_v1/OrbbecSDK_v1.10.37/SDK` 能看到就跳过）；换电脑时在**工程目录下**重新下载：
   ```bash
   cd /home/yunfeiwu/桌面/my_RL && mkdir -p vendor/orbbec_sdk_v1
   curl -L -o vendor/orbbec_sdk_v1/sdk.zip "https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.37/OrbbecSDK_C_C%2B%2B_v1.10.37_20260707_3f75820_linux_x64_release.zip"
   unzip -q vendor/orbbec_sdk_v1/sdk.zip -d vendor/orbbec_sdk_v1 && rm vendor/orbbec_sdk_v1/sdk.zip
   ```
   udev 规则：**本机已装**（`ls /etc/udev/rules.d/99-obsensor-libusb.rules` 能看到就跳过）。换电脑时（操作者执行，然后拔插相机）：
   ```bash
   cd /home/yunfeiwu/桌面/my_RL
   sudo cp vendor/orbbec_sdk_v1/OrbbecSDK_v1.10.37/Script/99-obsensor-libusb.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
3. 启动 CAN（操作者执行）：
   ```bash
   sudo ip link set dev can0 type can bitrate 1000000 && sudo ip link set dev can0 up
   ```
4. 进环境（**之后所有命令都在这个终端、这个目录里跑**；系统里没有 `python` 命令，只有激活 `.venv` 后才有）：
   ```bash
   cd /home/yunfeiwu/桌面/my_RL && source .venv/bin/activate
   ```

## 1. 检查设备 — step0

```bash
python -m nero_vs_rl.sim2real.calibration.step0_check_devices
```

应当看到（相机部分 2026-11-21 已在本机只读验证过）：

- 相机：DaBai DC1，序列号 CC1N16201EN；深度 640x400 Y11 + 彩色 1280x720 MJPG，约 24 帧/秒；出厂内参和深度→彩色外参都有。
  提示"连在 USB2.0 上"是正常的，不掉帧就不用管。
- 机械臂：关节角，以及 SDK 法兰位姿与本工程 MJCF 正解的比较——其中一个 body 的偏移应接近 0 或是一个小的常量。**机械臂读数这一部分还没在真机上跑过**，下午是第一次，有报错把输出发给我。
- `calib_data/*_check/` 里有 color.png / depth.png，看一眼彩色是否清楚、深度是否正常。

## 2. 手眼标定采样 — step1（约 20 分钟）

1. 标定板平放并用胶带固定在带面或桌面上，放在机械臂够得着的位置，避开强反光。**采样过程中不能碰动它。**
2. **夹爪张到最大**，否则手指会挡住标定板。
3. 运行：
   ```bash
   python -m nero_vs_rl.sim2real.calibration.step1_capture_handeye
   ```
   窗口里绿字表示正常。操作：摆好姿态 → 松手 → 等 `arm still: YES` → 按**空格**。`u` 撤销上一张，`q` 结束。
4. 姿态要求（决定标定质量）：
   - 采 **15～25 张**，每张都要看到**整块板**，距离 25～45 cm；
   - 从板的前、后、左、右各方向**倾斜 20°～40°** 去看，再加几张**绕相机光轴转 ±30°** 的；
   - 只平移不旋转没用：窗口里 `rot span` 要 ≥ 30°，`axis div` 越大越好（≥ 0.25）。

## 3. 求解 — step2（现场马上跑）

```bash
python -m nero_vs_rl.sim2real.calibration.step2_solve_handeye --all-sessions            # 合并全部真机采样（推荐）
python -m nero_vs_rl.sim2real.calibration.step2_solve_handeye --all-sessions --apply    # 通过后写入 dc1_site.yaml
```

**推荐多采几轮再合并**：每次 step1 是一个会话（15～25 张），采 3 轮以上后用 `--all-sessions` 合并（也可以 `--session 目录A 目录B …` 指定）。每个会话各算一个标定板位姿，所以两轮之间板被碰过也没关系；合并结果放在 `calib_data/<时间>_handeye_merged/`。

| 指标 | 合格线 | 说明 / 不合格怎么办 |
|---|---|---|
| 有效样本 | ≥ 12 | 补采 |
| 旋转跨度 | ≥ 30° | 补大角度倾斜的姿态 |
| **相机外参稳定性**：去掉一个会话（单会话时去掉 1/3 数据）重算，X 的最大变化 | ≤ 3 mm、≤ 0.5° | **判断标定结果能不能用的主指标**。不合格就再采一轮，合并 |
| 各样本推算的板位姿偏差 | > 15 mm 或 > 3° 判粗差 | 5 mm / 1° 以内最好；超过只是提示——这是机械臂名义运动学与真实位姿的差（绝对精度），多采也不会变小 |
| 基座系核对（SDK 法兰↔MJCF 是否固定变换） | 离散 ≤ 2 mm、0.2° | 运动学不一致，先查原因，别用于训练 |

> 2026-11-22 现场经验：单个会话的结果会因姿态不同相差 5～10 mm（一次 13 张的会话"通过"了旧标准，但三折稳定性是 7.4 mm）。6 个会话、107 张合并后稳定性 3.0 mm / 0.39°，各样本整链重投影约 6.6 px——这几个像素来自机械臂本身的绝对精度，而不是相机。

程序会自动剔除离群样本（最多 20%）和 PnP 误差 > 1 px 的图。也可以用 `--exclude 3 7` 手动排除某几张（合并后的编号）。默认用 DC1 出厂彩色内参（已用 112 张图自标核对过，几乎一致）。

## 4. 在仿真里核对 — step3

```bash
python -m nero_vs_rl.sim2real.calibration.step3_verify_in_sim --session calib_data/20261122_003203_handeye_merged
```

（`--session` 后面写 step2 打印的那个目录名，不要照抄尖括号。）把每张样本的实测关节角放进仿真环境，用仿真相机重新预测角点。**判定看"与 step2 同一指标的差异 < 0.5 px"**：说明标定结果放进仿真时坐标系转换、父 body、配置加载都没出错。预测误差本身会有几个像素（机械臂绝对误差），这不是 step3 要管的。对照图在 `<目录>/verify/`：左边是真机图（绿点为检测角点，红圈为仿真预测），右边是仿真腕部相机渲染（仿真里没有标定板，主要看手指位置是否一致）。

## 5. 测近距盲区 — step4（约 5 分钟）

```bash
python -m nero_vs_rl.sim2real.calibration.step4_measure_blind_zone --apply
```

哑光平板正对相机，从 ~40 cm 慢慢移近，直到画面中央深度变黑；来回做 2～3 次后按 `q`。结果写入 `camera.min_depth_m`（含 1 cm 余量）。

## 6. 测带的位置 — step5（约 5 分钟）

用示教把机械臂摆到带正上方，相机朝下、离带面约 40 cm，整条带宽都要在深度画面里；松手后运行：

```bash
python -m nero_vs_rl.sim2real.calibration.step5_measure_belt --handeye calib_data/20261122_003203_handeye_merged --apply
```

会输出带中心线 x、半宽、带面高度，并和 11-13 独立实测的带面平面对比：差值 < 1 cm 说明手眼标定和基座系都对。俯视图存在 `calib_data/*_belt/belt_top_view.png`（绿色是带面，红线是测出的两条边）。

已经测好、想事后再写入（或多次测量取平均、保留原带面高度）时，不用连设备：

```bash
python -m nero_vs_rl.sim2real.calibration.step5_measure_belt --from calib_data/20261122_004019_belt calib_data/20261122_004034_belt --apply --keep-z
```

> 2026-11-22：两次测量中心线 −0.3773/−0.3774 m、半宽 0.154/0.156 m（已写入）；带面测得 −0.112 m，比 11-13 独立实测高 10 mm，按 `--keep-z` 保留 −0.12 m（独立实测且经 NERO 真机抓取验证）。

## 7. 收尾

- 检查 `nero_vs_rl/algorithms/config/dc1_site.yaml`：里面应有 camera（位姿、内参、盲区）和 belt_* 三项，以及每一步的 provenance。
- 用真实相机参数确认环境仍然可解：
  ```bash
  python -m nero_vs_rl.algorithms.scripts.sac_eval --episodes 30      # 默认就用 dc1_site.yaml
  python -m nero_vs_rl.examples.view_wrist_camera --out /home/yunfeiwu/adddisk1/MY_RL_test/simulation/wrist_dc1   # 看仿真里的 DC1 视角
  ```
  如果脚本基线成功率明显下降，通常是真实安装位置让目标更早进入盲区或被手指挡住，需要先调整策略或观测再训练。
- 训练：`python -m nero_vs_rl.algorithms.scripts.sac_train --run-name sac_dc1_v2`（默认就用 dc1_site.yaml）

## 去实验室前可以先干跑一遍

不接任何设备，用仿真相机和仿真机械臂把 step1→2→3 跑通（step1 自动摆 20 个姿态）：

```bash
python -m nero_vs_rl.sim2real.calibration.step1_capture_handeye --sim --auto 20 --out runs/calib_dry
python -m nero_vs_rl.sim2real.calibration.step2_solve_handeye --session runs/calib_dry
python -m nero_vs_rl.sim2real.calibration.step3_verify_in_sim --session runs/calib_dry
```

干跑时**不要**加 `--apply`，免得把仿真数据写进 dc1_site.yaml。step2 会打印与仿真真值的误差（应 < 1 mm）。

## 故障排查

| 现象 | 处理 |
|---|---|
| step0 看不到相机 | 换 USB 口/线；`lsusb | grep 2bc5` 应有 0657（深度）和 0557（彩色）；装 udev 规则后拔插；确认 vendor/ 下有 SDK v1 |
| 提示 USB2.0 / 掉帧 | DC1 在 USB2.0 上也能出 640x400 深度 + 1280x720 MJPG 彩色（实测约 24 帧/秒）；掉帧就换 USB3 口 |
| 出厂标定记录不匹配 | 深度/彩色分辨率必须是出厂记录里的一对：640x400 + 1280x720，或 640x400/320x200/160x100 + 640x480 |
| 出厂内参/外参无效、"镜像"报错 | 驱动发现出厂标定不可用时会拒绝启动（不会带着错的内参继续）；记下 step0 输出发给我 |
| `arm still` 一直是 no | 关节读数抖动大于 3e-4 rad：确认已松手；必要时 `--hold 1.5` |
| 稳定性不合格 | 再采一轮（姿态尽量和前几轮不同），`--all-sessions` 合并 |
| 板位姿粗差（>15 mm） | 会话中板是否被碰、相机支架是否松动、是否都在完全静止时采样 |

## 文件

| 文件 | 作用 |
|---|---|
| `step0_check_devices.py` … `step5_measure_belt.py` | 按顺序运行的各步骤 |
| `board.py` | 棋盘格定义、角点检测（统一角点顺序，避免 180° 歧义）、PnP |
| `handeye.py` | AX=XB 求解、多会话（每会话一个板位姿）重投影联合优化、一致性与稳定性评估（纯数学，有离线测试） |
| `sim_bridge.py` | OpenCV↔MuJoCo 相机坐标转换，写 dc1_site.yaml |
| `common.py` | MJCF 正解、数据目录、验收阈值 |
| `../hardware/dc1_camera.py`、`../hardware/orbbec_v1.py` | DC1 驱动（Orbbec SDK v1.10.37，ctypes；深度+彩色同一帧组） |
| `../hardware/nero_arm.py` | NERO 只读反馈（关节角、SDK 法兰位姿、静止判定） |
| `../hardware/sim_devices.py` | 仿真相机/机械臂，用于干跑和测试 |
