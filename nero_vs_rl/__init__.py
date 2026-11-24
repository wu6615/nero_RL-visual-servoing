"""RL + eye-in-hand visual servoing for dynamic grasping on the AgileX NERO7 (GAP-RL layout)."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = PACKAGE_DIR / "assets"
SCENE_XML = ASSET_DIR / "descriptions" / "nero7" / "scene.xml"
CONFIG_DIR = PACKAGE_DIR / "algorithms" / "config"
# 训练日志、模型、评估视频等大文件放到数据盘（系统盘空间不够）；calib_data/ 仍在工程目录
SIM_OUTPUT_DIR = Path("/home/yunfeiwu/adddisk1/MY_RL_test/simulation")
# 【说明】训练 / 评估 / 看相机的默认配置 = 现场实测配置 dc1_site.yaml（base: default.yaml + 标定值）。
#   default.yaml 只剩占位值（75° 视场、0.15 m 盲区、带中心 −0.40），需要对照时显式 --config 指定。
#   dc1_site.yaml 不存在（还没标定）时退回 default.yaml。
SITE_CONFIG = CONFIG_DIR / "dc1_site.yaml"
TRAIN_CONFIG = SITE_CONFIG if SITE_CONFIG.exists() else CONFIG_DIR / "default.yaml"
# 真机录制（部署离线阶段）的 RGB-D + 关节数据也放数据盘，单次录制几百 MB
REAL_DATA_DIR = Path("/home/yunfeiwu/adddisk1/MY_RL_test/real")
