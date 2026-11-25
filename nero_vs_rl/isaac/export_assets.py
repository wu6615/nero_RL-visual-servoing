"""Run once on the development PC (needs MuJoCo): export kinematics table and a clean URDF for the server.

  python -m nero_vs_rl.isaac.export_assets

结果写进 isaac/assets/（几百 KB，随代码一起走）；机器人模型改了才需要重跑。只读模型，不跑仿真。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .. import ASSET_DIR, SCENE_XML
from . import KINEMATICS_NPZ, ROBOT_URDF

MESH_DIR = ASSET_DIR / "descriptions" / "nero7" / "meshes"
MIN_MASS, MIN_INERTIA = 0.01, 1e-5          # link7 的 1e-6 kg 对 PhysX 太小；抬到 10 g（夹爪总重 0.5 kg，影响可忽略）
EFFORT = {"gripper_joint1": 40.0, "gripper_joint2": 40.0}


def _mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _hull(v: np.ndarray) -> np.ndarray:
    return v[ConvexHull(v).vertices]


def export_kinematics(m: mujoco.MjModel) -> dict:
    R_fix, p_fix = [], []
    for i in range(1, 8):
        b = m.body(f"link{i}").id
        j = m.joint(f"joint{i}")
        assert np.allclose(j.axis, [0, 0, 1]) and np.allclose(j.pos, 0), "关节轴应为局部 z"
        R_fix.append(_mat(m.body_quat[b]))
        p_fix.append(m.body_pos[b])
    bl = m.body("base_link").id
    assert np.allclose(m.body_pos[bl], 0) and np.allclose(_mat(m.body_quat[bl]), np.eye(3)), "世界系应等于基座系"
    fl, gb = m.body("gripper_flange").id, m.body("gripper_base").id
    Rf, pf = _mat(m.body_quat[fl]), m.body_pos[fl]
    R_gb, p_gb = Rf @ _mat(m.body_quat[gb]), pf + Rf @ m.body_pos[gb]
    s = m.site("ee_site").id

    def body_verts(name, R=np.eye(3), p=np.zeros(3)):
        out = []
        b = m.body(name).id
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g]):
                k = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                out.append((v @ _mat(m.geom_quat[g]).T + m.geom_pos[g]) @ R.T + p)
        return _hull(np.concatenate(out))

    fingers, slides, planes = [], [], []
    for k, sign in ((1, 1.0), (2, -1.0)):                # 行程 q：joint1 = q，joint2 = −q（对称）
        b = m.body(f"gripper_link{k}").id
        R, p = _mat(m.body_quat[b]), m.body_pos[b].copy()
        v = body_verts(f"gripper_link{k}", R, p)
        fingers.append(v)
        slides.append(R @ m.joint(f"gripper_joint{k}").axis * sign)
        planes.append(np.unique(np.round(ConvexHull(v).equations, 9), axis=0))       # n·x + c ≤ 0 为内部；共面三角形合并
    joints = [m.joint(f"joint{i}").id for i in range(1, 8)]
    return dict(R_fix=np.array(R_fix), p_fix=np.array(p_fix), R_gb=R_gb, p_gb=p_gb, p_tcp=m.site_pos[s], R_tcp=_mat(m.site_quat[s]),
                q_lo=m.jnt_range[joints, 0], q_hi=m.jnt_range[joints, 1], verts_base=body_verts("gripper_base"),
                verts_f1=fingers[0], verts_f2=fingers[1], slide1=slides[0], slide2=slides[1],
                planes_f1=planes[0], planes_f2=planes[1])


# =============================================================================
# 【说明】为什么导出 URDF 而不是直接转 MJCF：URDF 导入器对关节、惯量、碰撞网格的处理最直接，body 名原样保留；
#   MJCF 里的执行器、等式约束、传感器 Isaac 用不上，驱动参数在 sim/scene.py 里按 Isaac 的口径写。
#   夹爪两指在 MuJoCo 里由等式约束对称；URDF 里是两个独立滑动关节，由环境发对称目标。
# =============================================================================
def export_urdf(m: mujoco.MjModel):
    robot = ET.Element("robot", name="nero7")
    bodies = ["base_link"] + [f"link{i}" for i in range(1, 8)] + ["gripper_flange", "gripper_base", "gripper_link1", "gripper_link2"]
    fmt = lambda v: " ".join(f"{x:.9g}" for x in v)
    origin = lambda parent, p, R: ET.SubElement(parent, "origin", xyz=fmt(p), rpy=fmt(Rotation.from_matrix(R).as_euler("xyz")))
    for name in bodies:
        b = m.body(name).id
        link = ET.SubElement(robot, "link", name=name)
        ine = ET.SubElement(link, "inertial")
        origin(ine, m.body_ipos[b], _mat(m.body_iquat[b]))
        ET.SubElement(ine, "mass", value=f"{max(float(m.body_mass[b]), MIN_MASS):.9g}")
        ix, iy, iz = np.maximum(m.body_inertia[b], MIN_INERTIA)
        ET.SubElement(ine, "inertia", ixx=f"{ix:.9g}", ixy="0", ixz="0", iyy=f"{iy:.9g}", iyz="0", izz=f"{iz:.9g}")
        file = {"gripper_base": "gripper_base_drawing_20261012"}.get(name, name)
        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag)
            g = ET.SubElement(el, "geometry")
            ET.SubElement(g, "mesh", filename=f"../../assets/descriptions/nero7/meshes/{file}.stl")
    for name in bodies[1:]:
        b = m.body(name).id
        parent = m.body(m.body_parentid[b]).name
        nj = m.body_jntnum[b]
        jname = m.joint(m.body_jntadr[b]).name if nj else f"{name}_fixed"
        kind = "fixed" if not nj else ("prismatic" if m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_SLIDE else "revolute")
        j = ET.SubElement(robot, "joint", name=jname, type=kind)
        origin(j, m.body_pos[b], _mat(m.body_quat[b]))
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=name)
        if nj:
            jid = m.body_jntadr[b]
            ET.SubElement(j, "axis", xyz=fmt(m.jnt_axis[jid]))
            lo, hi = m.jnt_range[jid]
            ET.SubElement(j, "limit", lower=f"{lo:.9g}", upper=f"{hi:.9g}", effort=f"{EFFORT.get(jname, 100.0):g}",
                          velocity="1.0" if kind == "prismatic" else "3.927")
    ET.indent(robot)
    ET.ElementTree(robot).write(ROBOT_URDF, encoding="utf-8", xml_declaration=True)


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    KINEMATICS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    k = export_kinematics(m)
    np.savez_compressed(KINEMATICS_NPZ, **k)
    export_urdf(m)
    print(f"{KINEMATICS_NPZ}  夹爪凸包顶点 {len(k['verts_base'])} + {len(k['verts_f1'])} + {len(k['verts_f2'])}\n{ROBOT_URDF}")


if __name__ == "__main__":
    main()
