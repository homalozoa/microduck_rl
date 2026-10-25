#!/usr/bin/env python3
"""Rebuild the training MJCFs from the vendored DuckURDF, preserving policy ABI.

The original Onshape MJCFs are read-only templates for joint/body frames,
sensors, actuator defaults and the optional roller attachments. Geometry,
joint origins/limits and rigid-body dynamics come from DuckURDF. Joint signs
are converted explicitly so existing task poses and the 14-channel deployment
interface retain their meaning. The previously unactuated jaw stays closed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "src/mjlab_microduck/robot"
SOURCE = ROBOT / "duckurdf"
TEMPLATE = ROBOT / "microduck"
VARIANTS = ("walk", "groundcontact", "allcollisions", "groundcontact_rollers")
# Position of URDF base_link in the historical trunk_base observation frame.
# HOME sole height: old 0.117128 m; DuckURDF 0.141028 m. A fixed reference
# translation preserves the existing height goals without rescaling the robot.
BASE_LINK_IN_TRUNK = np.array([0.0, 0.0, 0.024])
LINKS = {
    "base_link": "trunk_base",
    "neck_Link": "neck",
    "head_pitch_Link": "neck_pitch",
    "head_yaw_Link": "yaw_roll_motion",
    "head_roll_Link": "jaw_soft",
    "left_hip_yaw_Link": "yaw2roll",
    "left_hip_roll_Link": "hip_l",
    "left_hip_pitch_Link": "upper_leg_left",
    "left_knee_Link": "leg",
    "left_ankle_Link": "ankle_left",
    "right_hip_yaw_Link": "bearing_roll",
    "right_hip_roll_Link": "hip_l_2",
    "right_hip_pitch_Link": "upper_leg_right",
    "right_knee_Link": "leg_2",
    "right_ankle_Link": "ankle_right",
}


def vec(s: str) -> np.ndarray:
    return np.fromstring(s, sep=" ")


def fmt(v) -> str:
    return " ".join(f"{x:.16g}" for x in np.asarray(v).ravel())


def quat(rotation: np.ndarray) -> str:
    q = Rotation.from_matrix(rotation).as_quat()
    return fmt(q[[3, 0, 1, 2]])


def transform(el: ET.Element | None, *, urdf=False):
    if el is None:
        return np.eye(3), np.zeros(3)
    if urdf:
        return Rotation.from_euler("xyz", vec(el.get("rpy", "0 0 0"))).as_matrix(), vec(el.get("xyz", "0 0 0"))
    q = vec(el.get("quat", "1 0 0 0"))
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), vec(el.get("pos", "0 0 0"))


def template_frames(body, parent_rotation=np.eye(3), parent_pos=np.zeros(3)):
    r, p = transform(body)
    r, p = parent_rotation @ r, parent_pos + parent_rotation @ p
    out = {body.get("name"): (r, p)}
    for child in body.findall("body"):
        out.update(template_frames(child, r, p))
    return out


def vertices(link: str):
    data = (SOURCE / "meshes" / f"{link}.STL").read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    dtype = np.dtype([("normal", "<f4", 3), ("vertex", "<f4", (3, 3)), ("attr", "u2")])
    return np.frombuffer(data, dtype=dtype, count=n, offset=84)["vertex"].reshape(-1, 3).astype(float)


def inertial(link, rotation, translation):
    src = link.find("inertial")
    r, p = transform(src.find("origin"), urdf=True)
    a = src.find("inertia").attrib
    tensor = np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                       [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                       [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]])
    tensor = rotation @ r @ tensor @ r.T @ rotation.T
    return ET.Element("inertial", mass=src.find("mass").get("value"),
                      pos=fmt(translation + rotation @ p),
                      fullinertia=fmt([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                       tensor[0, 1], tensor[0, 2], tensor[1, 2]]))


def build(variant, links, joints, urdf_frames):
    xml = ET.parse(TEMPLATE / f"robot_{variant}.xml").getroot()
    xml.set("model", "microduck_duckurdf")
    xml.find("compiler").set("meshdir", "..")
    for mesh in xml.findall("asset/mesh"):
        mesh.set("file", "microduck/assets/" + mesh.get("file"))
    for link in links:
        ET.SubElement(xml.find("asset"), "mesh", name="duck_" + link,
                      file=f"duckurdf/meshes/{link}.STL")
    root = xml.find("worldbody/body")
    frames = template_frames(root)
    mapping = dict(LINKS)
    # Historical variants use different names for the same head/ankle frames.
    for link, names in {
        "head_roll_Link": ("jaw_soft", "bottom_head_shell"),
        "left_ankle_Link": ("ankle_left", "ankle_l_v1"),
        "right_ankle_Link": ("ankle_right", "ankle_r_v1"),
    }.items():
        mapping[link] = next(n for n in names if n in frames)
    bodies = {b.get("name"): b for b in xml.findall(".//worldbody//body")}
    new_positions = {mapping[l]: urdf_frames[l][1] + BASE_LINK_IN_TRUNK for l in mapping}
    new_positions["trunk_base"] = np.zeros(3)
    parents = {c: p for p in root.iter("body") for c in p.findall("body")}
    signs = {}
    for lname, bname in mapping.items():
        link, body = links[lname], bodies[bname]
        rbody, _ = frames[bname]
        rlink, plink = urdf_frames[lname]
        plink = plink + BASE_LINK_IN_TRUNK
        relative = rbody.T @ rlink
        translation = rbody.T @ (plink - new_positions[bname])
        parent = parents.get(body)
        if parent is not None:
            body.set("pos", fmt(frames[parent.get("name")][0].T @ (plink - new_positions[parent.get("name")])))
        for joint in body.findall("joint"):
            uname = "neck_joint" if joint.get("name") == "neck_pitch" else joint.get("name") + "_joint"
            src = joints[uname]
            source_axis = rlink @ vec(src.find("axis").get("xyz"))
            canonical_axis = rbody @ vec(joint.get("axis", "0 0 1"))
            sign = float(np.dot(source_axis, canonical_axis))
            assert np.isclose(abs(sign), 1), (uname, sign)
            sign = int(round(sign))
            signs[uname] = sign
            limit = src.find("limit")
            joint.set("range", fmt(sorted(sign * float(limit.get(k)) for k in ("lower", "upper"))))
            # BAM rewrites actuator forcerange from voltage, so put the URDF
            # effort ceiling on the joint's total actuator force as well.
            effort = float(limit.get("effort"))
            joint.set("actuatorfrclimited", "true")
            joint.set("actuatorfrcrange", fmt([-effort, effort]))
        # Keep only true servo-housing contact proxies in the protective-fall
        # variant. The source STL merges housings with links, so their separate
        # identity cannot be recovered from the URDF mesh.
        servo_proxies = []
        if variant == "allcollisions":
            for geom in body.findall("geom"):
                if geom.get("mesh") == "xl330" and geom.get("class") == "collision":
                    proxy = copy.deepcopy(geom)
                    proxy.set("name", f"{bname}_{len(servo_proxies)}_servo_collision")
                    servo_proxies.append(proxy)
        for el in list(body):
            if el.tag in ("inertial", "geom"):
                body.remove(el)
        body.insert(0, inertial(link, relative, translation))
        for kind in ("visual", "collision"):
            if kind == "collision" and (variant == "walk" or lname.endswith("ankle_Link")):
                continue
            src = link.find(kind)
            r, p = transform(src.find("origin"), urdf=True)
            attrs = dict(type="mesh", mesh="duck_" + lname, pos=fmt(translation + relative @ p),
                         quat=quat(relative @ r), **{"class": kind})
            if kind == "visual":
                color = src.find("material/color")
                attrs["rgba"] = color.get("rgba") if color is not None else "0.7 0.7 0.7 1"
            else:
                attrs["name"] = bname + "_collision"
            ET.SubElement(body, "geom", **attrs)
        body.extend(servo_proxies)
        if lname == "base_link":
            for site in body.findall("site"):
                site.set("pos", fmt(vec(site.get("pos", "0 0 0")) + translation))
            for proxy in servo_proxies:
                proxy.set("pos", fmt(vec(proxy.get("pos", "0 0 0")) + translation))
        if lname.endswith("ankle_Link"):
            side = lname.split("_")[0]
            v = vertices(lname)
            floor = v[:, 2].min()
            sole = v[v[:, 2] < floor + .002]
            lo, hi = sole.min(0), sole.max(0)
            center = (lo + hi) / 2
            center[2] = floor + .001
            site = body.find(f"site[@name='{side}_foot']")
            if "rollers" not in variant:
                # A thin sole box retains separate foot contact channels; a
                # convex hull of the whole ankle would also count shin hits.
                ET.SubElement(body, "geom", name=f"{side}_foot_collision", type="box",
                              pos=fmt(relative @ center), quat=quat(relative),
                              size=fmt([(hi[0]-lo[0])/2, (hi[1]-lo[1])/2, .001]),
                              **{"class": "collision"})
                center[2] = floor
                site.set("pos", fmt(relative @ center))
    # Merge the closed jaw into the head rigid body, just like the old model.
    # A fixed MJCF body is fused by mjlab; its mass and inertia are retained.
    head = bodies[mapping["head_roll_Link"]]
    rhead = frames[mapping["head_roll_Link"]][0]
    rjaw, pjaw = urdf_frames["jaw_Link"]
    rel = rhead.T @ rjaw
    delta = rhead.T @ (pjaw + BASE_LINK_IN_TRUNK - new_positions[mapping["head_roll_Link"]])
    jaw = ET.SubElement(head, "body", name="duckurdf_closed_jaw", pos=fmt(delta), quat=quat(rel))
    jaw.append(inertial(links["jaw_Link"], np.eye(3), np.zeros(3)))
    ET.SubElement(jaw, "geom", type="mesh", mesh="duck_jaw_Link", **{"class": "visual"}, rgba="0.9 0.5 0.2 1")
    if variant != "walk":
        ET.SubElement(jaw, "geom", name="jaw_collision", type="mesh", mesh="duck_jaw_Link", **{"class": "collision"})
    mouth = head.find("site[@name='mouth_tip']")
    if mouth is not None:
        v = vertices("jaw_Link")
        tip = v[np.argmax(v[:, 0])].copy()
        tip[1] = 0
        mouth.set("pos", fmt(delta + rel @ tip))
    # Merged CAD hulls overlap at these mating assemblies, despite the links
    # being separated by an intermediate joint. Do not turn that approximation
    # into permanent self-contact forces/reward penalties at HOME.
    contact = xml.find("contact")
    if contact is None:
        contact = ET.SubElement(xml, "contact")
    for a, b in (("trunk_base", "hip_l"), ("trunk_base", "hip_l_2"),
                 ("neck_pitch", mapping["head_roll_Link"]),
                 ("neck_pitch", "duckurdf_closed_jaw")):
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    ET.indent(xml, space="  ")
    out = SOURCE / f"robot_{variant}.xml"
    ET.ElementTree(xml).write(out, encoding="utf-8", xml_declaration=True)
    return signs


def main():
    raw = SOURCE / "urdf/duckurdf.urdf"
    source = ET.parse(raw).getroot()
    links = {l.get("name"): l for l in source.findall("link")}
    joints = {j.get("name"): j for j in source.findall("joint")}
    frames = {"base_link": (np.eye(3), np.zeros(3))}
    pending = list(joints.values())
    while pending:
        for j in pending[:]:
            parent = j.find("parent").get("link")
            if parent not in frames:
                continue
            r, p = transform(j.find("origin"), urdf=True)
            rp, pp = frames[parent]
            frames[j.find("child").get("link")] = (rp @ r, pp + rp @ p)
            pending.remove(j)
        if pending and not any(j.find("parent").get("link") in frames for j in pending):
            raise ValueError("Disconnected URDF tree")
    signs = {v: build(v, links, joints, frames) for v in VARIANTS}
    # Use the same proven gearbox transformation as the original variants.
    import subprocess
    for v in VARIANTS:
        out = SOURCE / f"robot_{v}_backlash.xml"
        out.write_bytes((SOURCE / f"robot_{v}.xml").read_bytes())
        subprocess.run(["python3", str(TEMPLATE / "add_backlash.py"), str(out)], check=True)
    manifest = dict(source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                    total_mass_kg=sum(float(l.find("inertial/mass").get("value")) for l in links.values()),
                    jaw="fixed at URDF zero; unactuated as in original model",
                    base_link_in_trunk_base_m=BASE_LINK_IN_TRUNK.tolist(),
                    joint_effort_limits_Nm={n: float(j.find("limit").get("effort")) for n, j in joints.items()},
                    joint_velocity_ratings_rad_s={n: float(j.find("limit").get("velocity")) for n, j in joints.items()},
                    velocity_note="URDF ratings retained here; speed follows the unchanged BAM voltage/back-EMF model, not a nonphysical qvel clamp",
                    urdf_to_policy_joint_signs=signs,
                    collision_notes="URDF link convex hulls; separate sole boxes; legacy servo proxies and roller attachments")
    (SOURCE / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
