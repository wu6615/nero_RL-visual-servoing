"""DaBai DC1 RGB-D camera through the vendored Orbbec SDK v1 (depth + color as one SDK frameset).

Synchronous ``read()`` is enough for calibration (the arm is stationary while
sampling).  Intrinsics and the depth->color extrinsic come from the factory
calibration record that exactly matches the active depth/color sizes, and are
saved with every calibration session.
Ported from nero_conveyor_ros2 (git 327234a^, runtime/backends/dabai.py).
"""
from __future__ import annotations

import ctypes as ct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .orbbec_v1 import BackendError, BackendUnavailable, LegacyOrbbecSdk

# =============================================================================
# 【说明】DC1 只能用 Orbbec SDK v1（见 orbbec_v1.py），深度和彩色都由 SDK 出成一组帧。
#   2026-11-21 本机实测的模式（USB2.0 连接）：
#     深度 Y11/Y12：640x400@5/10/15/30，320x200，160x100，1280x800@7
#     彩色 RGB/BGRA/MJPG/YUYV：1280x720@≤30，640x480@≤30
#     出厂标定记录：深度 640x400 + 彩色 1280x720 有一条完全匹配（另有 640x480 彩色的几条）
#   默认取 640x400 Y11 深度 + 1280x720 MJPG 彩色（USB2 带宽紧，彩色用压缩格式）。
#   出厂内参只对"完全相同分辨率"的那一对有效，所以分辨率不匹配时直接报错，不做缩放。
# 坐标约定：OpenCV 相机系（x 右、y 下、z 朝前），单位米。
#   T_color_depth：深度相机系 → 彩色相机系，p_color = R p_depth + t（SDK 平移是 mm，这里转成 m）
# =============================================================================

PID_DC1_DEPTH = 0x0657
SENSOR_COLOR, SENSOR_DEPTH = 2, 3
FORMATS = {"YUYV": 0, "MJPG": 5, "Y16": 8, "Y11": 11, "Y12": 12, "RGB": 22, "BGR": 23, "BGRA": 25, "RGBA": 31}
FORMAT_NAMES = {v: k for k, v in FORMATS.items()}
# 镜像/翻转属性：出厂标定描述的是不镜像的光学坐标，开着的话临时关掉，stop() 时恢复
MIRROR_PROPS = {"OB_PROP_DEPTH_MIRROR_BOOL": 14, "OB_PROP_DEPTH_FLIP_BOOL": 15,
                "OB_PROP_COLOR_MIRROR_BOOL": 81, "OB_PROP_COLOR_FLIP_BOOL": 82}


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=lambda: [0.0] * 5)   # OpenCV 顺序 k1 k2 p1 p2 k3 [k4 k5 k6]
    model: str = "brown_conrady"

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def D(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=float)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(**d)


@dataclass
class RGBDFrame:
    color: np.ndarray | None     # (H, W, 3) uint8 RGB
    depth: np.ndarray | None     # (H, W) float32 米，0 = 无效
    stamp: float                 # 本机 monotonic 时间（收到帧时）


@dataclass
class DC1Config:
    serial: str | None = None
    depth_width: int = 640
    depth_height: int = 400
    depth_fps: int = 30
    depth_format: str = "Y11"
    color_width: int = 1280
    color_height: int = 720
    color_fps: int = 30
    color_format: str = "MJPG"
    timeout_ms: int = 1000
    min_depth_m: float = 0.02        # 只去掉明显的无效值；真实盲区由 step4 测
    max_depth_m: float = 5.0
    sdk_dir: str | None = None       # None = vendor/orbbec_sdk_v1/OrbbecSDK_v1.10.37/SDK


def _intrinsics(native, distortion, source: str) -> Intrinsics:
    # SDK v1 的畸变字段顺序是 k1..k6, p1, p2；转成 OpenCV 顺序
    k = [float(getattr(distortion, n)) for n in ("k1", "k2", "k3", "k4", "k5", "k6")]
    p = [float(distortion.p1), float(distortion.p2)]
    dist = [k[0], k[1], p[0], p[1], k[2]] + (k[3:] if any(abs(v) > 0 for v in k[3:]) else [])
    return Intrinsics(int(native.width), int(native.height), float(native.fx), float(native.fy),
                      float(native.cx), float(native.cy), dist, source)


def _geometry(param, depth_size, color_size) -> tuple[Intrinsics, Intrinsics, np.ndarray]:
    """Validate one factory CameraParam record against the active sizes."""
    if param.isMirrored:
        raise BackendUnavailable("出厂标定记录是镜像的，不能描述光学坐标")
    depth = _intrinsics(param.depthIntrinsic, param.depthDistortion, "factory")
    color = _intrinsics(param.rgbIntrinsic, param.rgbDistortion, "factory")
    for name, intr, size in (("深度", depth, depth_size), ("彩色", color, color_size)):
        if (intr.width, intr.height) != tuple(size):
            raise BackendUnavailable(f"{name}出厂内参是 {intr.width}x{intr.height}，与当前 {size[0]}x{size[1]} 不符")
        if intr.fx <= 1 or not (0 <= intr.cx <= intr.width and 0 <= intr.cy <= intr.height):
            raise BackendUnavailable(f"{name}出厂内参无效：{intr}")
    T = np.eye(4)
    T[:3, :3] = np.asarray(param.transform.rot, float).reshape(3, 3)
    T[:3, 3] = np.asarray(param.transform.trans, float) / 1000.0
    R = T[:3, :3]
    if not (np.allclose(R.T @ R, np.eye(3), atol=2e-5) and np.isclose(np.linalg.det(R), 1, atol=2e-5)):
        raise BackendUnavailable("深度→彩色出厂外参不是刚体变换")
    if not 0.001 <= np.linalg.norm(T[:3, 3]) <= 0.3:
        raise BackendUnavailable("深度→彩色出厂基线缺失或不合理")
    return depth, color, T


def decode_depth(payload: bytes, width: int, height: int, format_id: int, scale_mm: float,
                 min_m: float = 0.02, max_m: float = 5.0) -> np.ndarray:
    """SDK unpacked Y11/Y12/Y16 -> metres (0 = invalid)."""
    if format_id not in (8, 11, 12) or len(payload) != width * height * 2:
        raise BackendUnavailable(f"深度帧格式/大小不对：format={format_id}，{len(payload)} 字节")
    if not np.isfinite(scale_mm) or scale_mm <= 0:
        raise BackendUnavailable(f"深度刻度无效：{scale_mm}")
    depth = np.frombuffer(payload, dtype="<u2").reshape(height, width).astype(np.float32) * (scale_mm / 1000.0)
    depth[(depth < min_m) | (depth > max_m)] = 0
    return depth


def decode_rgb(payload: bytes, width: int, height: int, format_id: int) -> np.ndarray:
    import cv2
    if format_id == 5:
        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (height, width, 3):
            raise BackendUnavailable("MJPG 彩色帧解码失败或尺寸不符")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    channels = 4 if format_id in (25, 31) else 3
    if format_id in (22, 23, 25, 31) and len(payload) == width * height * channels:
        img = np.frombuffer(payload, np.uint8).reshape(height, width, channels)
        if format_id in (23, 25):
            img = img[..., [2, 1, 0]]
        return img[..., :3].copy()
    if format_id == 0 and len(payload) == width * height * 2:
        return cv2.cvtColor(np.frombuffer(payload, np.uint8).reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUY2)
    raise BackendUnavailable(f"不支持的彩色帧格式 {FORMAT_NAMES.get(format_id, format_id)}")


class _Session:
    """Context + device list + the single DC1 device; closes in reverse order."""

    def __init__(self, sdk: LegacyOrbbecSdk):
        self.sdk = sdk
        self.context = sdk.call("ob_create_context_with_config", str(sdk.config_path).encode())
        self.devices = sdk.call("ob_query_device_list", self.context)
        self.device = self.info = 0

    def dc1_indices(self) -> list[int]:
        n = self.sdk.call("ob_device_list_device_count", self.devices)
        return [i for i in range(n) if self.sdk.call("ob_device_list_get_device_pid", self.devices, i) == PID_DC1_DEPTH]

    def open(self, index: int) -> dict:
        sdk = self.sdk
        self.device = sdk.call("ob_device_list_get_device", self.devices, index)
        self.info = sdk.call("ob_device_get_device_info", self.device)
        text = lambda name: sdk.call(name, self.info).decode("utf-8", errors="replace")  # noqa: E731
        return dict(name=text("ob_device_info_name"), serial=text("ob_device_info_serial_number"),
                    pid=hex(sdk.call("ob_device_info_pid", self.info)),
                    firmware=text("ob_device_info_firmware_version"),
                    connection=text("ob_device_info_connection_type"))

    def close(self):
        for name, attr in (("ob_delete_device_info", "info"), ("ob_delete_device", "device"),
                           ("ob_delete_device_list", "devices"), ("ob_delete_context", "context")):
            self.sdk.close_handle(name, getattr(self, attr))
            setattr(self, attr, 0)


def list_devices(sdk_dir: str | None = None) -> list[dict]:
    """All connected DC1s (opens each briefly to read name/serial/firmware; no streaming)."""
    s = _Session(LegacyOrbbecSdk(sdk_dir))
    try:
        out = []
        for i in s.dc1_indices():
            out.append(s.open(i))
            s.sdk.close_handle("ob_delete_device_info", s.info)
            s.sdk.close_handle("ob_delete_device", s.device)
            s.info = s.device = 0
        return out
    finally:
        s.close()


class DC1Camera:
    def __init__(self, cfg: DC1Config | None = None):
        self.cfg = cfg or DC1Config()
        self.sdk: LegacyOrbbecSdk | None = None
        self._s: _Session | None = None
        self._pipeline = self._config = 0
        self._profiles: list[int] = []
        self._prop_originals: dict[str, bool] = {}
        self.color_intrinsics: Intrinsics | None = None
        self.depth_intrinsics: Intrinsics | None = None
        self.T_color_depth: np.ndarray | None = None
        self.device_info: dict = {}
        self.active_profiles: dict = {}
        self.calibration_source: str | None = None
        self.warnings: list[str] = []

    # ---- 打开 / 关闭
    def _select(self, sensor: int, prefix: str) -> int:
        sdk, c = self.sdk, self.cfg
        w, h, fps, fmt = (getattr(c, f"{prefix}_{k}") for k in ("width", "height", "fps", "format"))
        profiles = sdk.call("ob_pipeline_get_stream_profile_list", self._pipeline, sensor)
        try:
            try:
                p = sdk.call("ob_stream_profile_list_get_video_stream_profile", profiles, w, h, FORMATS[fmt.upper()], fps)
            except BackendError as e:
                raise BackendUnavailable(f"DC1 没有 {prefix} 模式 {w}x{h} {fmt}@{fps}；可用模式见 list_profiles()") from e
            self._profiles.append(p)
            self.active_profiles[prefix] = dict(width=w, height=h, fps=fps, format=fmt.upper())
            return p
        finally:
            sdk.close_handle("ob_delete_stream_profile_list", profiles)

    def _read_calibration(self):
        sdk, c = self.sdk, self.cfg
        dsz, csz = (c.depth_width, c.depth_height), (c.color_width, c.color_height)
        try:
            param = sdk.call("ob_pipeline_get_camera_param_with_profile", self._pipeline, *csz, *dsz)
            geo = _geometry(param, dsz, csz)
            self.calibration_source = "sdk_matched_profiles"
        except (BackendError, ValueError) as first:
            # 旧后端经验：DC1 上按 profile 取标定有时返回空，退回设备出厂记录里找完全匹配的一条
            recs = sdk.call("ob_device_get_calibration_camera_param_list", self._s.device)
            try:
                matches = []
                for i in range(sdk.call("ob_camera_param_list_count", recs)):
                    cand = sdk.call("ob_camera_param_list_get_param", recs, i)
                    if ((cand.depthIntrinsic.width, cand.depthIntrinsic.height) == dsz
                            and (cand.rgbIntrinsic.width, cand.rgbIntrinsic.height) == csz):
                        matches.append((i, cand))
                if len(matches) != 1:
                    raise BackendUnavailable(f"深度 {dsz} + 彩色 {csz} 需要恰好一条出厂标定记录，找到 {len(matches)} 条"
                                             f"（按 profile 取标定也失败：{first}）")
                idx, param = matches[0]
                geo = _geometry(param, dsz, csz)
                self.calibration_source = f"device_factory_record_{idx}"
            finally:
                sdk.close_handle("ob_delete_camera_param_list", recs)
        self.depth_intrinsics, self.color_intrinsics, self.T_color_depth = geo

    def _bool_prop(self, name: str) -> bool | None:
        sdk, dev, pid = self.sdk, self._s.device, MIRROR_PROPS[name]
        try:
            if not sdk.has_function("ob_device_is_property_supported") or \
                    not sdk.call("ob_device_is_property_supported", dev, pid, 1):
                return None
            return bool(sdk.call("ob_device_get_bool_property", dev, pid))
        except BackendError:
            return None

    def _disable_mirrors(self):
        for name, pid in MIRROR_PROPS.items():
            if self._bool_prop(name):
                try:
                    self.sdk.call("ob_device_set_bool_property", self._s.device, pid, False)
                    self._prop_originals[name] = True
                    self.warnings.append(f"{name} 原来是开的，已临时关闭（stop 时恢复）")
                except BackendError as e:
                    raise BackendUnavailable(f"关不掉 {name}，出厂标定与图像方向不符：{e}") from e

    def start(self) -> "DC1Camera":
        try:
            self.sdk = LegacyOrbbecSdk(self.cfg.sdk_dir)
            self._s = _Session(self.sdk)
            idx = self._s.dc1_indices()
            if not idx:
                raise BackendUnavailable("没找到 DC1（PID 0x0657）：检查 USB 和 udev 规则，见 calibration/README.md")
            chosen = None
            for i in idx:
                info = self._s.open(i)
                if self.cfg.serial is None or info["serial"] == self.cfg.serial:
                    chosen = info
                    break
                self.sdk.close_handle("ob_delete_device_info", self._s.info)
                self.sdk.close_handle("ob_delete_device", self._s.device)
                self._s.info = self._s.device = 0
            if chosen is None:
                raise BackendUnavailable(f"没找到序列号 {self.cfg.serial} 的 DC1；在线：{list_devices(self.cfg.sdk_dir)}")
            if len(idx) > 1 and self.cfg.serial is None:
                self.warnings.append(f"接了 {len(idx)} 台 DC1，用了第一台；用 serial 指定")
            self.device_info = chosen
            if "2.0" in chosen.get("connection", ""):
                self.warnings.append("DC1 连在 USB2.0 上：带宽有限，如有掉帧请换 USB3 口")
            sdk = self.sdk
            self._pipeline = sdk.call("ob_create_pipeline_with_device", self._s.device)
            dp = self._select(SENSOR_DEPTH, "depth")
            cp = self._select(SENSOR_COLOR, "color")
            self._read_calibration()
            self._disable_mirrors()
            self._config = sdk.call("ob_create_config")
            for p in (dp, cp):
                sdk.call("ob_config_enable_stream", self._config, p)
            sdk.call("ob_config_set_align_mode", self._config, 0)                    # 不做 SDK 对齐，保留原生像素
            sdk.call("ob_config_set_frame_aggregate_output_mode", self._config, 0)  # 只要深度+彩色齐全的帧组
            sdk.call("ob_pipeline_start_with_config", self._pipeline, self._config)
            return self
        except Exception:
            self.stop()
            raise

    def stop(self):
        sdk = self.sdk
        if sdk is None:
            return
        if self._pipeline:
            sdk.close_handle("ob_pipeline_stop", self._pipeline)
        if self._s is not None and self._s.device:
            for name, value in self._prop_originals.items():
                try:
                    sdk.call("ob_device_set_bool_property", self._s.device, MIRROR_PROPS[name], value)
                except BackendError as e:
                    self.warnings.append(f"恢复 {name} 失败：{e}")
        self._prop_originals.clear()
        for p in self._profiles:
            sdk.close_handle("ob_delete_stream_profile", p)
        self._profiles.clear()
        sdk.close_handle("ob_delete_config", self._config)
        sdk.close_handle("ob_delete_pipeline", self._pipeline)
        self._config = self._pipeline = 0
        if self._s is not None:
            self._s.close()
            self._s = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- 取帧
    def read(self, timeout_ms: int | None = None, need_color: bool = True) -> RGBDFrame:
        """Next complete depth+color frameset (SDK pairs them; no hardware sync)."""
        sdk, c = self.sdk, self.cfg
        if not self._pipeline:
            raise BackendUnavailable("DC1 没有在出流：先 start()")
        t_ms = timeout_ms or c.timeout_ms
        deadline = time.monotonic() + 5 * t_ms / 1000
        while time.monotonic() < deadline:
            fs = depth = color = 0
            try:
                fs = sdk.call("ob_pipeline_wait_for_frameset", self._pipeline, t_ms)
                if not fs:
                    continue
                depth = sdk.call("ob_frameset_depth_frame", fs)
                color = sdk.call("ob_frameset_color_frame", fs)
                if not depth or (need_color and not color):
                    continue
                stamp = time.monotonic()
                d = self._decode(depth, "depth")
                rgb = self._decode(color, "color") if color else None
                return RGBDFrame(rgb, d, stamp)
            finally:
                for h in (depth, color, fs):
                    if h:
                        sdk.close_handle("ob_delete_frame", h)
        raise TimeoutError("DC1 取帧超时")

    def _decode(self, frame, kind: str):
        sdk, c = self.sdk, self.cfg
        w, h = (int(sdk.call(n, frame)) for n in ("ob_video_frame_width", "ob_video_frame_height"))
        want = (c.depth_width, c.depth_height) if kind == "depth" else (c.color_width, c.color_height)
        if (w, h) != want:
            raise BackendUnavailable(f"{kind} 帧尺寸变成 {w}x{h}，与出厂标定 {want} 不符")
        payload = ct.string_at(sdk.call("ob_frame_data", frame), sdk.call("ob_frame_data_size", frame))
        fmt = sdk.call("ob_frame_format", frame)
        if kind == "depth":
            return decode_depth(payload, w, h, fmt, float(sdk.call("ob_depth_frame_get_value_scale", frame)),
                                c.min_depth_m, c.max_depth_m)
        return decode_rgb(payload, w, h, fmt)

    # ---- 诊断
    def list_profiles(self) -> dict[str, list[str]]:
        """Available modes per sensor (needs start(), or opens a temporary pipeline)."""
        own = not self._pipeline
        if own:
            self.sdk = self.sdk or LegacyOrbbecSdk(self.cfg.sdk_dir)
            self._s = _Session(self.sdk)
            idx = self._s.dc1_indices()
            if not idx:
                self._s.close()
                raise BackendUnavailable("没找到 DC1（PID 0x0657）")
            self._s.open(idx[0])
            pipe = self.sdk.call("ob_create_pipeline_with_device", self._s.device)
        else:
            pipe = self._pipeline
        sdk, out = self.sdk, {}
        try:
            for sensor, name in ((SENSOR_DEPTH, "DEPTH"), (SENSOR_COLOR, "COLOR"), (1, "IR")):
                try:
                    pl = sdk.call("ob_pipeline_get_stream_profile_list", pipe, sensor)
                except BackendError:
                    continue
                items = set()
                for i in range(sdk.call("ob_stream_profile_list_count", pl)):
                    p = sdk.call("ob_stream_profile_list_get_profile", pl, i)
                    fmt = sdk.call("ob_stream_profile_format", p)
                    items.add((sdk.call("ob_video_stream_profile_width", p), sdk.call("ob_video_stream_profile_height", p),
                               FORMAT_NAMES.get(fmt, str(fmt)), sdk.call("ob_video_stream_profile_fps", p)))
                    sdk.close_handle("ob_delete_stream_profile", p)
                sdk.close_handle("ob_delete_stream_profile_list", pl)
                out[name] = [f"{w}x{h} {f}@{fps}" for w, h, f, fps in sorted(items)]
        finally:
            if own:
                sdk.close_handle("ob_delete_pipeline", pipe)
                self._s.close()
                self._s = None
        return out

    def camera_info(self) -> dict:
        return dict(device=self.device_info, profiles=self.active_profiles,
                    color_intrinsics=self.color_intrinsics.to_dict() if self.color_intrinsics else None,
                    depth_intrinsics=self.depth_intrinsics.to_dict() if self.depth_intrinsics else None,
                    T_color_depth=None if self.T_color_depth is None else self.T_color_depth.tolist(),
                    calibration_source=self.calibration_source,
                    sdk=f"Orbbec SDK v{self.sdk.version if self.sdk else '1'}", warnings=list(self.warnings))
