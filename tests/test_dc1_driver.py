"""DC1 SDK v1 driver pieces that need no hardware: ABI, decoding, factory-calibration checks."""
import ctypes as ct

import cv2
import numpy as np
import pytest

from nero_vs_rl.sim2real.hardware.dc1_camera import _geometry, _intrinsics, decode_depth, decode_rgb
from nero_vs_rl.sim2real.hardware.orbbec_v1 import (BackendUnavailable, OBCameraParam, OBDistortion,
                                                    OBIntrinsic)


def _param(dw=640, dh=400, cw=1280, ch=720, baseline_mm=-29.9, mirrored=False):
    p = OBCameraParam()
    p.depthIntrinsic = OBIntrinsic(475.2, 475.2, 323.9, 199.0, dw, dh)
    p.rgbIntrinsic = OBIntrinsic(903.1, 903.1, 644.3, 306.5, cw, ch)
    p.rgbDistortion = OBDistortion(0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.01, 0.02)
    p.transform.rot[:] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    p.transform.trans[:] = [baseline_mm, 0, 0]
    p.isMirrored = mirrored
    return p


def test_packed_abi_matches_sdk_headers():
    assert ct.sizeof(OBCameraParam) == 153


def test_distortion_reordered_to_opencv():
    intr = _intrinsics(_param().rgbIntrinsic, _param().rgbDistortion, "factory")
    assert intr.dist == pytest.approx([0.1, -0.2, 0.01, 0.02, 0.3])      # k1 k2 p1 p2 k3


def test_geometry_accepts_matching_record_and_converts_mm():
    depth, color, T = _geometry(_param(), (640, 400), (1280, 720))
    assert (depth.width, color.width) == (640, 1280)
    assert T[0, 3] == pytest.approx(-0.0299)


@pytest.mark.parametrize("kwargs,sizes", [
    ({}, ((640, 400), (640, 480))),                  # 彩色分辨率与记录不符
    ({"mirrored": True}, ((640, 400), (1280, 720))),
    ({"baseline_mm": 0.0}, ((640, 400), (1280, 720))),
])
def test_geometry_rejects_bad_records(kwargs, sizes):
    with pytest.raises(BackendUnavailable):
        _geometry(_param(**kwargs), *sizes)


def test_decode_depth_scales_and_masks():
    raw = np.array([[0, 1000], [3000, 60000]], dtype="<u2")
    d = decode_depth(raw.tobytes(), 2, 2, 11, 1.0, min_m=0.02, max_m=5.0)
    assert np.allclose(d, [[0.0, 1.0], [3.0, 0.0]], atol=1e-6)


def test_decode_rgb_mjpg_and_bgra():
    img = np.zeros((8, 16, 3), np.uint8)
    img[..., 0] = 200                                                   # 红色（RGB）
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out = decode_rgb(jpg.tobytes(), 16, 8, 5)
    assert out.shape == (8, 16, 3) and abs(int(out[..., 0].mean()) - 200) < 5
    bgra = np.zeros((8, 16, 4), np.uint8)
    bgra[..., 2] = 200
    assert decode_rgb(bgra.tobytes(), 16, 8, 25)[..., 0].mean() == 200
