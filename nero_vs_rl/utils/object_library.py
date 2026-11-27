"""Real workpiece library: fit box/cylinder primitives to the STL print files and load the result for the envs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .. import ASSET_DIR

# =============================================================================
# 【说明】现场工件 → 仿真工件库（2026-11-26，用户给了 20 个打印用 STL，单位 mm，见 docs/现场几何来源.md）
#   部署感知把工件拟合成长方体或圆柱（perception.py），策略看到的抓取也是按这两种基本形状解析生成的。
#   所以仿真不直接用网格，而是用"真实工件平放时的拟合长方体 / 圆柱"：和真机感知给策略的输入一致。
#   每个 STL：
#     1. 平放姿态：凸包上每个平面当底面，体积重心投影落在底面多边形内才算放得稳；
#        按 底面积 × 重心离边距离 / 重心高度 排序，取最稳的那个（人放到带上通常就是这样放）
#     2. 平放后：高 = 竖直方向跨度；俯视轮廓求最小外接矩形 → 窄边（夹持方向）、长边
#     3. 俯视轮廓圆度 4πA/P² ≥ 0.97 记为圆柱（直径 = 窄边），否则记为长方体
#     4. 窄边超过夹爪可用开口（0.008 + 2 × 0.05 − 0.012 = 96 mm，同 dynamic_grasp.max_width）的工件夹不了，
#        记在 excluded 里，不进训练
#   库文件只存尺寸（几 KB，放在 assets/objects/）；STL 本身 100 多 MB，留在数据盘。
#   生成：python -m nero_vs_rl.utils.object_library /home/yunfeiwu/adddisk1/MY_RL_test/real/objects_stl
# =============================================================================
LIBRARY_DIR = ASSET_DIR / "objects"
CIRCLE_MIN = 0.97


def load_stl(path: str | Path) -> np.ndarray:
    """Triangles (N, 3, 3) in the file's units (binary or ASCII STL)."""
    b = Path(path).read_bytes()
    n = int.from_bytes(b[80:84], "little") if len(b) >= 84 else -1
    if n >= 0 and len(b) == 84 + 50 * n:                                       # 二进制：长度对得上就按二进制读
        rec = np.frombuffer(b[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        return rec["v"].astype(float)
    v = [list(map(float, ln.split()[1:4])) for ln in b.decode(errors="ignore").splitlines() if ln.strip().startswith("vertex")]
    return np.asarray(v, float).reshape(-1, 3, 3)


def volume_and_centroid(T: np.ndarray) -> tuple[float, np.ndarray]:
    """Signed-tetrahedron volume and centroid of a closed triangle mesh."""
    dv = np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])) / 6.0
    vol = float(dv.sum())
    return abs(vol), (dv[:, None] * T.sum(1) / 4.0).sum(0) / vol


def _min_rect(P2: np.ndarray) -> tuple[float, float, float]:
    """Smallest-area enclosing rectangle of 2D points (rotating calipers): (short, long, circularity of the hull)."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(P2)
    H, best = P2[h.vertices], None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        e = e / np.linalg.norm(e)
        a, b = np.ptp(H @ e), np.ptp(H @ np.array([-e[1], e[0]]))
        if best is None or a * b < best[0]:
            best = (a * b, min(a, b), max(a, b))
    return best[1], best[2], float(4 * np.pi * h.volume / h.area ** 2)    # 二维 ConvexHull：volume = 面积，area = 周长


def rest_poses(T: np.ndarray) -> list[dict]:
    """Stable resting faces, most stable first: dict(up, width, length, height, circularity, area, margin)."""
    from scipy.spatial import ConvexHull
    V = np.unique(T.reshape(-1, 3), axis=0)
    _, com = volume_and_centroid(T)
    hull = ConvexHull(V)
    planes: dict[tuple, list] = {}
    for eq, simp in zip(hull.equations, hull.simplices):                     # 共面的凸包三角形合成一个面
        planes.setdefault(tuple(np.round(eq, 3)), []).append(simp)
    out = []
    for key, simps in planes.items():
        n = np.asarray(key[:3]) / np.linalg.norm(key[:3])                    # 外法向 = 放下后的竖直向下
        up = -n
        u = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        w = np.cross(up, u)
        pts = V[np.unique(np.concatenate(simps))]
        if len(pts) < 3:
            continue
        try:
            face = ConvexHull(np.c_[pts @ u, pts @ w])
        except Exception:                                                     # 退化（共线）
            continue
        margin = -(face.equations[:, :2] @ np.array([com @ u, com @ w]) + face.equations[:, 2]).max()
        if margin <= 0 or face.volume <= 0:
            continue                                                          # 重心不在底面上：放不稳
        Vr = np.c_[V @ u, V @ w, V @ up]
        width, length, circ = _min_rect(Vr[:, :2])
        com_h = com @ up - Vr[:, 2].min()
        out.append(dict(up=up, width=width, length=length, height=float(np.ptp(Vr[:, 2])), circularity=circ,
                        area=float(face.volume), margin=float(margin), score=float(face.volume * margin / com_h)))
    return sorted(out, key=lambda r: -r["score"])


def fit_object(T: np.ndarray, max_width: float) -> dict:
    """Primitive for the most stable resting pose (T in metres)."""
    r = rest_poses(T)[0]
    vol, _ = volume_and_centroid(T)
    kind = "cylinder" if r["circularity"] >= CIRCLE_MIN else "box"
    size = [r["width"]] * 2 + [r["height"]] if kind == "cylinder" else [r["width"], r["length"], r["height"]]
    return dict(type=kind, size_m=[round(float(s), 4) for s in size], volume_cm3=round(vol * 1e6, 1),
                rest_area_cm2=round(r["area"] * 1e4, 1), graspable=bool(r["width"] <= max_width))


def build_library(stl_dir: str | Path, max_width: float, unit: float = 1e-3) -> dict:
    objects, excluded = [], []
    for p in sorted(Path(stl_dir).glob("*.stl")):
        e = dict(name=p.stem, **fit_object(load_stl(p) * unit, max_width))
        (objects if e.pop("graspable") else excluded).append(e)
    return dict(source=str(stl_dir), max_width_m=round(max_width, 4), objects=objects, excluded=excluded)


# ----------------------------------------------------------------------------- 环境侧
_CACHE: dict[Path, list[dict]] = {}


def library_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else LIBRARY_DIR / p


def load_library(name: str, types: list[str] | None = None) -> list[dict]:
    """Graspable entries of a library file (optionally only the given primitive types)."""
    p = library_path(name)
    if p not in _CACHE:
        _CACHE[p] = yaml.safe_load(p.read_text(encoding="utf-8"))["objects"]
    lib = [e for e in _CACHE[p] if types is None or e["type"] in types]
    if not lib:
        raise ValueError(f"工件库 {p} 里没有类型为 {types} 的工件")
    return lib


def sample_half(entry: dict, rng, noise: float) -> np.ndarray:
    """Half sizes (box: hx hy hz；cylinder: r r h) with ±noise relative jitter (cylinder radius jittered once)."""
    s = np.asarray(entry["size_m"], float) / 2.0
    k = 1.0 + rng.uniform(-noise, noise, 3) if noise > 0 else np.ones(3)
    if entry["type"] == "cylinder":
        k[1] = k[0]
    return s * k


def main(argv=None):
    ap = argparse.ArgumentParser(description="从工件 STL 生成仿真工件库（拟合长方体 / 圆柱）")
    ap.add_argument("stl_dir")
    ap.add_argument("--out", default=str(LIBRARY_DIR / "real_parts.yaml"))
    ap.add_argument("--unit", type=float, default=1e-3, help="STL 单位换算到米（默认 mm）")
    ap.add_argument("--max-width", type=float, default=0.096, help="夹爪可用开口（m）")
    args = ap.parse_args(argv)
    lib = build_library(args.stl_dir, args.max_width, args.unit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head = ("# 现场工件库：由 nero_vs_rl/utils/object_library.py 从打印用 STL 生成，不要手改。\n"
            "# size_m：长方体 = 窄边（夹持方向）、长边、高；圆柱 = 直径、直径、高。都是最稳平放姿态下的外接尺寸。\n")
    out.write_text(head + yaml.safe_dump(lib, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for e in lib["objects"]:
        print(f"  {e['name']}  {e['type']:8s} {' × '.join(f'{s * 1000:.0f}' for s in e['size_m'])} mm")
    for e in lib["excluded"]:
        print(f"  {e['name']}  夹不了：窄边 {e['size_m'][0] * 1000:.0f} mm > {args.max_width * 1000:.0f} mm")
    print(f"已写入 {out}：{len(lib['objects'])} 个可夹，{len(lib['excluded'])} 个排除")
    return lib


if __name__ == "__main__":
    main()
