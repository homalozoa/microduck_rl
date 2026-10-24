"""Analytic, differentiable re-implementation of the Mjlab-Velocity-Flat-MicroDuck rewards.

Each function mirrors one reward term of ``make_microduck_velocity_env_cfg`` and returns
the UNWEIGHTED term value (N,) exactly as the env's term function does; ``VelocityReward``
applies the live weights from the env's reward manager (so curricula carry over) and the
step_dt scale. Inputs are plain tensors (``Kin`` / ``Contacts``) so the same code runs on
the env's own quantities (unit tests, tests/test_supermimic_rewards.py) and on
world-model states (imagination).

``soft=True`` (imagination only) replaces the hard indicators of the contact-timing terms
with sigmoids of width ``tau`` so the policy gets a gradient; ``soft=False`` is the exact
env formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab_microduck.supermimic.rotations import projected_gravity

TERMS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
    "body_ang_vel",
    "angular_momentum",
    "dof_pos_limits",
    "action_rate_l2",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "self_collisions",
    "head_pose_tracking",
    "body_pose_tracking",
    "head_pose_bias",
)


@dataclass
class Kin:
    quat_w: torch.Tensor  # (N,4) trunk (root link) orientation
    lin_vel_b: torch.Tensor  # (N,3) root link linear velocity, body frame
    ang_vel_b: torch.Tensor  # (N,3)
    ang_vel_w: torch.Tensor  # (N,3) trunk angular velocity, world frame
    joint_pos: torch.Tensor  # (N,NJ)
    foot_height: torch.Tensor  # (N,2) site z above the (flat) ground
    foot_vel_xy: torch.Tensor  # (N,2,2) foot site linear velocity xy, world
    angmom: torch.Tensor  # (N,3) subtree angular momentum of the root


@dataclass
class Contacts:
    contact: torch.Tensor  # (N,2) foot contact (bool->float, or probability)
    air_time: torch.Tensor  # (N,2) current air time after this step (s)
    first_contact: torch.Tensor  # (N,2) contact established within this step
    peak_height: torch.Tensor  # (N,2) swing peak height incl. this step (before landing reset)
    self_collision: torch.Tensor  # (N,) number of self-contacts (or probability)


@dataclass
class Commands:
    twist: torch.Tensor  # (N,3)
    head: torch.Tensor  # (N,4)
    body: torch.Tensor  # (N,6)


def _active(twist: torch.Tensor, threshold: float) -> torch.Tensor:
    total = torch.norm(twist[:, :2], dim=1) + torch.abs(twist[:, 2])
    return (total > threshold).float()


def _ind(x: torch.Tensor, soft: bool, tau: float) -> torch.Tensor:
    """1[x > 0] or sigmoid(x / tau)."""
    return torch.sigmoid(x / tau) if soft else (x > 0).float()


class VelocityReward:
    """Reads term parameters + live weights from an env's reward manager."""

    def __init__(self, env, soft_tau_time: float = 0.02, soft_tau_height: float = 0.002):
        rm = env.reward_manager
        self.rm = rm
        self.dt = env.step_dt
        robot = env.scene["robot"]
        p = {n: rm.get_term_cfg(n).params for n in rm.active_terms}
        self.params = p
        self.names = tuple(rm.active_terms)
        assert set(self.names) <= set(TERMS), set(self.names) - set(TERMS)
        dev = env.device
        self.q0 = robot.data.default_joint_pos[0].clone()
        self.soft_limits = robot.data.soft_joint_pos_limits[0].clone()  # (NJ,2)
        # pose: leg joints + per-regime std tensors (resolved by the env's term class)
        pose_cfg = rm.get_term_cfg("pose")
        self.pose_ids = torch.tensor(pose_cfg.params["asset_cfg"].joint_ids, device=dev, dtype=torch.long) if not isinstance(pose_cfg.params["asset_cfg"].joint_ids, slice) else torch.arange(len(robot.joint_names), device=dev)
        pose_impl = pose_cfg.func
        self.pose_std = (pose_impl.std_standing, pose_impl.std_walking, pose_impl.std_running)
        ids, _ = robot.find_joints_by_actuator_names([r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"])
        self.neck_ids = torch.tensor(ids, device=dev, dtype=torch.long)
        self.tau_t = soft_tau_time
        self.tau_h = soft_tau_height

    def weights(self) -> dict[str, float]:
        return {n: float(self.rm.get_term_cfg(n).weight) for n in self.names}

    # ------------------------------------------------------------------ terms

    def terms(self, k: Kin, c: Contacts, cmd: Commands, action, prev_action, head_ema, soft: bool = False) -> dict[str, torch.Tensor]:
        p = self.params
        out: dict[str, torch.Tensor] = {}
        tw = cmd.twist
        # track_linear_velocity / track_angular_velocity
        std = p["track_linear_velocity"]["std"]
        err = torch.sum(torch.square(tw[:, :2] - k.lin_vel_b[:, :2]), dim=1) + torch.square(k.lin_vel_b[:, 2])
        out["track_linear_velocity"] = torch.exp(-err / std**2)
        std = p["track_angular_velocity"]["std"]
        err = torch.square(tw[:, 2] - k.ang_vel_b[:, 2]) + torch.sum(torch.square(k.ang_vel_b[:, :2]), dim=1)
        out["track_angular_velocity"] = torch.exp(-err / std**2)
        # upright (trunk_base == root link)
        g = projected_gravity(k.quat_w)
        out["upright"] = torch.exp(-torch.sum(torch.square(g[:, :2]), dim=1) / p["upright"]["std"] ** 2)
        # pose (variable_posture)
        pp = p["pose"]
        speed = torch.norm(tw[:, :2], dim=1) + torch.abs(tw[:, 2])
        stand = (speed < pp["walking_threshold"]).float()
        walk = ((speed >= pp["walking_threshold"]) & (speed < pp["running_threshold"])).float()
        run = (speed >= pp["running_threshold"]).float()
        s0, s1, s2 = self.pose_std
        std = s0 * stand[:, None] + s1 * walk[:, None] + s2 * run[:, None]
        e2 = torch.square(k.joint_pos[:, self.pose_ids] - self.q0[self.pose_ids])
        out["pose"] = torch.exp(-torch.mean(e2 / std**2, dim=1))
        # body_ang_vel (trunk, world xy), angular_momentum
        out["body_ang_vel"] = torch.sum(torch.square(k.ang_vel_w[:, :2]), dim=1)
        out["angular_momentum"] = torch.sum(torch.square(k.angmom), dim=-1)
        # dof_pos_limits
        lo, hi = self.soft_limits[:, 0], self.soft_limits[:, 1]
        out["dof_pos_limits"] = torch.sum(-(k.joint_pos - lo).clip(max=0.0) + (k.joint_pos - hi).clip(min=0.0), dim=1)
        # action_rate_l2
        out["action_rate_l2"] = torch.sum(torch.square(action - prev_action), dim=1)
        # air_time
        pa = p["air_time"]
        at = c.air_time
        in_range = _ind(at - pa["threshold_min"], soft, self.tau_t) * _ind(pa["threshold_max"] - at, soft, self.tau_t)
        out["air_time"] = in_range.sum(1) * _active(tw, pa["command_threshold"])
        # foot_clearance
        pc = p["foot_clearance"]
        vel = torch.norm(k.foot_vel_xy, dim=-1)
        out["foot_clearance"] = torch.sum(torch.abs(k.foot_height - pc["target_height"]) * vel, dim=1) * _active(tw, pc["command_threshold"])
        # foot_swing_height (cost at landing)
        ps = p["foot_swing_height"]
        e = c.peak_height / ps["target_height"] - 1.0
        out["foot_swing_height"] = torch.sum(torch.square(e) * c.first_contact, dim=1) * _active(tw, ps["command_threshold"])
        # foot_slip
        pf = p["foot_slip"]
        out["foot_slip"] = torch.sum(torch.square(vel) * c.contact, dim=1) * _active(tw, pf["command_threshold"])
        # self_collisions
        out["self_collisions"] = c.self_collision
        # head_pose_tracking
        ph = p["head_pose_tracking"]
        err = (k.joint_pos[:, self.neck_ids] - self.q0[self.neck_ids]) - cmd.head
        per = torch.exp(-((err / ph["std"]) ** 2))
        if ph.get("fine_std") is not None:
            fw = ph.get("fine_weight", 0.5)
            per = (1.0 - fw) * per + fw * torch.exp(-((err / ph["fine_std"]) ** 2))
        out["head_pose_tracking"] = per.mean(-1)
        # body_pose_tracking: weight 0 in Velocity-Flat (mjlab skips weight-0 terms)
        out["body_pose_tracking"] = torch.zeros_like(out["upright"])
        # head_pose_bias (head_ema already updated with this step's error)
        out["head_pose_bias"] = -head_ema.abs().mean(dim=-1)
        return {n: out[n] for n in self.names}

    def head_error(self, joint_pos: torch.Tensor, head_cmd: torch.Tensor) -> torch.Tensor:
        return (joint_pos[:, self.neck_ids] - self.q0[self.neck_ids]) - head_cmd

    def head_alpha(self) -> float:
        tau = self.params["head_pose_bias"].get("tau_s", 1.0)
        return min(1.0, self.dt / max(tau, 1e-6))

    def total(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum of weight * value * dt (mjlab RewardManager.compute; weight-0 terms skipped)."""
        w = self.weights()
        r = None
        for n, v in terms.items():
            if w[n] == 0.0:
                continue
            r = v * (w[n] * self.dt) if r is None else r + v * (w[n] * self.dt)
        return r
