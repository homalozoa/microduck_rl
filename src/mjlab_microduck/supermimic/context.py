"""Domain-randomization context xi read from a live env.

Everything the velocity cfg randomizes that changes the dynamics or the observation is
read back from where mjlab / BAM store it, so the world model sees the exact per-env
values:

    foot friction (1), BAM friction_scale (1), BAM vin (1), BAM vin drop gain (1),
    BAM kp_scale (1), BAM kd_scale (1), armature scale per servo dof (NJ),
    trunk CoM offset (3), head-assembly CoM offsets (5 x 3), trunk mass scale (1),
    encoder bias (NJ), IMU misalignment rotation vector (3)

The BAM command delay (3-6 physics steps) is NOT in xi (mjlab keeps the lag inside the
actuator's delay buffer); the world model gets the previous two actions instead.
"""

from __future__ import annotations

import math

import torch

from mjlab_microduck.supermimic.rotations import rotvec_from_quat

HEAD_BODY_REGEX = ("neck", "neck_pitch", "yaw_roll_motion", "(bottom_head_shell|jaw_soft)", "bearing_roll")


class ContextReader:
    def __init__(self, env):
        self.env = env
        robot = env.scene["robot"]
        self.robot = robot
        m = env.sim.model
        mj = env.sim.mj_model
        self.nj = len(robot.joint_names)
        self.trunk_id = int(robot.indexing.body_ids[robot.find_bodies(("trunk_base",))[0][0]])
        head_local = robot.find_bodies(HEAD_BODY_REGEX)[0]
        self.head_ids = [int(robot.indexing.body_ids[i]) for i in head_local]
        assert len(self.head_ids) == 5, self.head_ids
        geom_ids = [mj.geom("robot/" + n).id for n in ("left_foot_collision", "right_foot_collision")]
        self.foot_geom = geom_ids[0]
        self.dof_ids = robot.indexing.joint_v_adr.long()
        # Compile-time defaults (dr ops are "add"/"scale" relative to these).
        self.ipos0_trunk = torch.as_tensor(mj.body_ipos[self.trunk_id], dtype=torch.float32, device=env.device)
        self.ipos0_head = torch.as_tensor(mj.body_ipos[self.head_ids], dtype=torch.float32, device=env.device)
        self.mass0_trunk = float(mj.body_mass[self.trunk_id])
        self.arm0 = torch.as_tensor(mj.dof_armature[self.dof_ids.cpu().numpy()], dtype=torch.float32, device=env.device)
        self.actuator = robot.actuators[0]
        self.dim = 6 + self.nj + 3 + 15 + 1 + self.nj + 3

    @staticmethod
    def _per_env(x: torch.Tensor, n: int) -> torch.Tensor:
        return x if x.shape[0] == n else x.expand(n, *x.shape[1:])

    def read(self) -> torch.Tensor:
        env = self.env
        n = env.num_envs
        m = env.sim.model
        a = self.actuator
        fr = self._per_env(m.geom_friction, n)[:, self.foot_geom, 0:1]
        fs = a.friction_scale if hasattr(a, "friction_scale") else torch.ones(n, 1, device=env.device)
        vin = a.vin_tensor.reshape(n, -1)[:, :1]
        vdg = a.vin_drop_gain.reshape(n, -1)[:, :1] if a.vin_drop_gain is not None else torch.zeros(n, 1, device=env.device)
        kp = a.kp_scale.reshape(n, -1)[:, :1]
        kd = a.kd_scale.reshape(n, -1)[:, :1]
        arm = self._per_env(m.dof_armature, n)[:, self.dof_ids] / self.arm0.clamp_min(1e-9)
        ipos = self._per_env(m.body_ipos, n)
        com_t = ipos[:, self.trunk_id] - self.ipos0_trunk
        com_h = (ipos[:, self.head_ids] - self.ipos0_head).reshape(n, 15)
        mass = self._per_env(m.body_mass, n)[:, self.trunk_id : self.trunk_id + 1] / self.mass0_trunk
        bias = self.robot.data.encoder_bias
        bias = self._per_env(bias, n) if bias is not None else torch.zeros(n, self.nj, device=env.device)
        q_imu = getattr(env, "_imu_misalign_quat", None)
        imu = rotvec_from_quat(q_imu) if q_imu is not None else torch.zeros(n, 3, device=env.device)
        return torch.cat([fr, fs, vin, vdg, kp, kd, arm, com_t, com_h, mass, bias, imu], dim=-1).float()

    @staticmethod
    def imu_quat(xi: torch.Tensor) -> torch.Tensor:
        from mjlab_microduck.supermimic.rotations import quat_from_rotvec

        return quat_from_rotvec(xi[..., -3:])


def xi_slices(nj: int) -> dict[str, slice]:
    o = 6 + nj
    return {
        "foot_friction": slice(0, 1),
        "trunk_com": slice(o, o + 3),
        "head_com": slice(o + 3, o + 18),
        "trunk_mass": slice(o + 18, o + 19),
        "encoder_bias": slice(o + 19, o + 19 + nj),
        "imu_rotvec": slice(o + 19 + nj, o + 22 + nj),
    }


class XiNormalizer:
    """(xi - mean) / std, statistics frozen from the first batch of real contexts.

    Constant dims (e.g. kp_scale when kp DR is off) get std 1 so they map to 0.
    """

    def __init__(self, xi_batch: torch.Tensor):
        self.mean = xi_batch.mean(0)
        std = xi_batch.std(0)
        self.std = torch.where(std < 1e-6, torch.ones_like(std), std)

    def __call__(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d):
        self.mean, self.std = d["mean"], d["std"]


IMU_MAX_RAD = math.radians(6.0)
