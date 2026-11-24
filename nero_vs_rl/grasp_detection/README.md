# grasp_detection

GAP-RL 中 `localgrasp/`（LoG 抓取检测）的位置。

现在用 `analytic_grasps.py`：按盒子/圆柱尺寸解析生成平行夹爪抓取（物体系，TCP 约定：+z 接近、y 闭合、TCP 在指尖平面后 10 mm）。环境每步只把它们变换到当前位姿，再经过 `sensors/camera.py` 的可见性判断和噪声，所以不需要 GAP-RL 那样的预采数据集。

## 换成点云抓取检测

接口只需要输出 `GraspSet`：`T_obj (M,4,4)` 抓取位姿、`vis_obj (M,3)` 可见点、`scores (M,)`。

- 仿真训练：可以沿用 GAP-RL 的做法，离线在多视角下跑检测器、存成数据集，训练时按视角查表（`pick_single.py::_compute_near_grasps`）。
- 真机：直接在 DC1 点云上实时检测，输出 TCP 系抓取，跳过可见性模拟。
- GAP-RL 自带的 `LoG.so` 是 Cython 编译的闭源模块，按其 README 需要 Python 3.9 + torch 1.13 + pytorch3d，和本工程环境（3.10 / torch 2.14）不兼容，需要单独环境或换检测器。
