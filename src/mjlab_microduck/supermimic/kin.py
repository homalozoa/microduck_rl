"""World-model state <-> env, and the differentiable kinematics the rewards need.

State layout (MuJoCo, robot only, env-origin relative):
    s = [root_pos(3), root_quat wxyz(4), joint_pos(NJ), root_lin_vel_w(3), root_ang_vel_b(3), joint_vel(NJ)]
i.e. ``qpos | qvel`` of the free-floating robot (the walk model has no passive joints).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from mjlab_microduck.supermimic.context import xi_slices
from mjlab_microduck.supermimic.fk import MjKinematics
from mjlab_microduck.supermimic.rotations import matrix_from_quat, quat_apply, quat_apply_inverse, quat_mul
from mjlab_microduck.supermimic.terms import Kin

FOOT_SITES = ("left_foot", "right_foot")


def robot_mj_model(env) -> mujoco.MjModel:
    """Robot-only MjModel (free root first) compiled from the entity's own spec."""
    spec = env.scene["robot"].spec.copy()
    return spec.compile()


def read_state(env) -> torch.Tensor:
    """(N, 7+NJ+6+NJ) robot qpos|qvel with the env origin removed (call when derived data is fresh)."""
    r = env.scene["robot"]
    d = env.sim.data
    qpos = d.qpos[:, r.indexing.free_joint_q_adr] if hasattr(r.indexing, "free_joint_q_adr") else None
    if qpos is None:
        raise RuntimeError("robot indexing has no free_joint_q_adr")
    qpos = qpos.clone()
    qpos[:, 0:3] -= env.scene.env_origins
    jq = d.qpos[:, r.indexing.joint_q_adr]
    qvel = d.qvel[:, r.indexing.free_joint_v_adr]
    jv = d.qvel[:, r.indexing.joint_v_adr]
    return torch.cat([qpos, jq, qvel, jv], dim=-1).float()


class RobotKinematics:
    def __init__(self, env, device):
        mj = robot_mj_model(env)
        self.mj = mj
        self.fk = MjKinematics(mj, device=device)
        self.fk.body_names = [n.split("/")[-1] for n in self.fk.body_names]  # entity spec prefixes "robot/"
        pre = "robot/" if mj.nsite and mj.site(0).name.startswith("robot/") else ""
        self.nj = mj.nq - 7
        self.nq = mj.nq
        self.device = device
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        self.site_body = [self.fk.body_index(mj.body(int(mj.site(pre + s).bodyid[0])).name.split("/")[-1]) for s in FOOT_SITES]
        self.site_pos = t(np.stack([mj.site(pre + s).pos for s in FOOT_SITES]))  # (2,3) in body frame
        self.site_quat = t(np.stack([mj.site(pre + s).quat for s in FOOT_SITES]))
        self.mass0 = t(mj.body_mass[1:])  # (nb,)
        self.ipos0 = t(mj.body_ipos[1:])
        self.iquat = t(mj.body_iquat[1:])
        self.inertia0 = t(mj.body_inertia[1:])  # (nb,3) principal
        names = self.fk.body_names
        self.trunk = names.index("trunk_base")
        head = []
        for alt in ("neck", "neck_pitch", "yaw_roll_motion", ("bottom_head_shell", "jaw_soft"), "bearing_roll"):
            alts = alt if isinstance(alt, tuple) else (alt,)
            head.append(next(names.index(a) for a in alts if a in names))
        self.head = head
        self.sl = xi_slices(self.nj)

    def body_params(self, xi: torch.Tensor):
        """Per-env masses / CoM offsets / principal inertia from the context."""
        n = xi.shape[:-1]
        mass = self.mass0.expand(*n, -1).clone()
        ipos = self.ipos0.expand(*n, -1, -1).clone()
        inertia = self.inertia0.expand(*n, -1, -1).clone()
        s = xi[..., self.sl["trunk_mass"]]
        mass[..., self.trunk] = mass[..., self.trunk] * s[..., 0]
        inertia[..., self.trunk, :] = inertia[..., self.trunk, :] * s
        ipos[..., self.trunk, :] = ipos[..., self.trunk, :] + xi[..., self.sl["trunk_com"]]
        hc = xi[..., self.sl["head_com"]].reshape(*n, 5, 3)
        for j, b in enumerate(self.head):
            ipos[..., b, :] = ipos[..., b, :] + hc[..., j, :]
        return mass, ipos, inertia

    def kin(self, s: torch.Tensor, xi: torch.Tensor) -> Kin:
        nq = self.nq
        out = self.fk(s[..., :nq], s[..., nq:])
        pos, quat, lv, av = out["pos"], out["quat"], out["lin_vel"], out["ang_vel"]
        root_q = quat[..., 0, :]
        # feet
        fp, fv = [], []
        for i, b in enumerate(self.site_body):
            r = quat_apply(quat[..., b, :], self.site_pos[i].expand_as(pos[..., b, :]))
            fp.append(pos[..., b, :] + r)
            fv.append(lv[..., b, :] + torch.cross(av[..., b, :], r, dim=-1))
        fp = torch.stack(fp, -2)
        fv = torch.stack(fv, -2)
        # subtree angular momentum about the subtree CoM (MuJoCo subtreeangmom of the root)
        mass, ipos, inertia = self.body_params(xi)
        rc = quat_apply(quat, ipos)  # (…,nb,3) com offset in world
        com = pos + rc
        vcom = lv + torch.cross(av, rc, dim=-1)
        M = mass.sum(-1, keepdim=True)
        C = (mass[..., None] * com).sum(-2) / M
        V = (mass[..., None] * vcom).sum(-2) / M
        L = (mass[..., None] * torch.cross(com - C[..., None, :], vcom - V[..., None, :], dim=-1)).sum(-2)
        qi = quat_mul(quat, self.iquat.expand_as(quat))
        w_i = quat_apply_inverse(qi, av)
        L = L + quat_apply(qi, inertia * w_i).sum(-2)
        return Kin(
            quat_w=root_q,
            lin_vel_b=quat_apply_inverse(root_q, s[..., nq : nq + 3]),
            ang_vel_b=s[..., nq + 3 : nq + 6],
            ang_vel_w=av[..., 0, :],
            joint_pos=s[..., 7:nq],
            foot_height=fp[..., 2].clamp_min(0.0),  # ray sensor reports 0 for a penetrating foot
            foot_vel_xy=fv[..., :2],
            angmom=L,
        )


def env_kin(env, foot_height: torch.Tensor | None = None) -> Kin:
    """The env's own quantities, exactly as its reward functions read them (for unit tests)."""
    r = env.scene["robot"]
    d = r.data
    site_ids = r.find_sites(FOOT_SITES)[0]
    trunk = r.find_bodies(("trunk_base",))[0]
    return Kin(
        quat_w=d.root_link_quat_w,
        lin_vel_b=d.root_link_lin_vel_b,
        ang_vel_b=d.root_link_ang_vel_b,
        ang_vel_w=d.body_link_ang_vel_w[:, trunk, :].squeeze(1),
        joint_pos=d.joint_pos,
        foot_height=foot_height if foot_height is not None else env.scene["foot_height_scan"].data.heights,
        foot_vel_xy=d.site_lin_vel_w[:, site_ids, :2],
        angmom=env.scene["robot/root_angmom"].data,
    )


__all__ = ["RobotKinematics", "read_state", "env_kin", "robot_mj_model", "matrix_from_quat"]
