"""Physical equivalence to source URDF, and the existing 14-servo contract."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROBOT = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot"
MODEL = ROBOT / "duckurdf"
SERVO_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw",
    "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
SIGNS = dict(zip(SERVO_ORDER, [-1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1]))
BODY = {
    "base_link": "trunk_base", "neck_Link": "neck", "head_pitch_Link": "neck_pitch",
    "head_yaw_Link": "yaw_roll_motion", "head_roll_Link": "jaw_soft",
    "jaw_Link": "duckurdf_closed_jaw", "left_hip_yaw_Link": "yaw2roll",
    "left_hip_roll_Link": "hip_l", "left_hip_pitch_Link": "upper_leg_left",
    "left_knee_Link": "leg", "left_ankle_Link": "ankle_left",
    "right_hip_yaw_Link": "bearing_roll", "right_hip_roll_Link": "hip_l_2",
    "right_hip_pitch_Link": "upper_leg_right", "right_knee_Link": "leg_2",
    "right_ankle_Link": "ankle_right",
}


@pytest.mark.parametrize("path", sorted(MODEL.glob("robot_*.xml")))
def test_every_variant_keeps_servo_abi_mass_and_passives(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    names = [m.joint(int(j)).name for j in m.actuator_trnid[:, 0]]
    assert names == SERVO_ORDER
    assert [m.joint(j).name for j in range(1, m.njnt)
            if not m.joint(j).name.startswith("passive_")] == SERVO_ORDER
    assert m.nu == 14
    for name in SERVO_ORDER:
        j = m.joint(name).id
        assert m.jnt_actfrclimited[j]
        np.testing.assert_allclose(m.jnt_actfrcrange[j], [-.96, .96])
    assert not any("jaw" in m.joint(j).name for j in range(m.njnt))
    assert m.body_mass.sum() == pytest.approx(.784 + (.01789604 if "rollers" in path.name else 0))
    if "backlash" in path.name:
        assert sum(m.joint(j).name.endswith("_backlash") for j in range(m.njnt)) == 14
    if "rollers" in path.name:
        assert sum(m.joint(j).name.endswith("wheel") for j in range(m.njnt)) == 4


@pytest.mark.parametrize("path", sorted(MODEL.glob("robot_*.xml")))
def test_home_has_no_permanent_assembly_self_contacts(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    home = dict(zip(SERVO_ORDER, [0, -.0873, -.4579, -.0049, .453,
                                  .3491, .3491, 0, 0, 0, .0873, .4579, .0049, -.453]))
    for name, value in home.items():
        d.qpos[m.joint(name).qposadr[0]] = value
    mujoco.mj_forward(m, d)
    assert d.ncon == 0


def test_mjcf_matches_urdf_forward_kinematics_com_inertias_and_limits():
    src = ET.parse(MODEL / "urdf/duckurdf.urdf").getroot()
    joints = list(src.findall("joint"))
    m = mujoco.MjModel.from_xml_path(str(MODEL / "robot_groundcontact.xml"))
    d = mujoco.MjData(m)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        angles = dict(zip(SERVO_ORDER, rng.uniform(-.2, .2, 14)))
        for name, angle in angles.items():
            d.qpos[m.joint(name).qposadr[0]] = angle
        mujoco.mj_forward(m, d)
        # Source base_link sits 24 mm above the historical trunk reference.
        frames = {"base_link": (np.eye(3), np.array([0, 0, .144]))}
        pending = joints.copy()
        while pending:
            for j in pending[:]:
                parent = j.find("parent").get("link")
                if parent not in frames:
                    continue
                name = j.get("name")
                canonical = "neck_pitch" if name == "neck_joint" else name.removesuffix("_joint")
                angle = angles.get(canonical, 0) * SIGNS.get(canonical, 1)
                origin = j.find("origin")
                r = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
                p = np.fromstring(origin.get("xyz"), sep=" ")
                axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
                rp, pp = frames[parent]
                frames[j.find("child").get("link")] = (rp @ r @ Rotation.from_rotvec(axis * angle).as_matrix(), pp + rp @ p)
                if canonical in SIGNS:
                    limits = sorted(SIGNS[canonical] * float(j.find("limit").get(k)) for k in ("lower", "upper"))
                    np.testing.assert_allclose(m.jnt_range[m.joint(canonical).id], limits, atol=1e-12)
                pending.remove(j)
        for link in src.findall("link"):
            r, p = frames[link.get("name")]
            body_name = BODY[link.get("name")]
            b = m.body(body_name).id
            i = link.find("inertial")
            com = np.fromstring(i.find("origin").get("xyz"), sep=" ")
            np.testing.assert_allclose(d.xipos[b], p + r @ com, atol=2e-8)
            assert m.body_mass[b] == pytest.approx(float(i.find("mass").get("value")), rel=1e-12)
            a = i.find("inertia").attrib
            tensor = np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])],
                               [float(a['ixy']), float(a['iyy']), float(a['iyz'])],
                               [float(a['ixz']), float(a['iyz']), float(a['izz'])]])
            rotation = d.ximat[b].reshape(3, 3)
            np.testing.assert_allclose(rotation @ np.diag(m.body_inertia[b]) @ rotation.T,
                                       r @ tensor @ r.T, atol=3e-10)  # MuJoCo inertia eigensolver tolerance.
