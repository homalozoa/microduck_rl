"""Analytic 61-D actor observation of Mjlab-Velocity-Flat-MicroDuck from a world-model state.

Layout (env order, checked by tests/test_supermimic_obs.py):
    base_ang_vel(3) projected_gravity(3) joint_pos(14) joint_vel(14) actions(14)
    command(3) head_command(4) body_command(6)

Matches the env cfg: IMU misalignment rotation on ang vel + gravity, encoder bias on
joint_pos (joint_pos + bias - default), joint_vel lagged one control step, uniform
sensor noise. The env's 0-1 step random IMU delay is reproduced by an optional per-env
choice between the current and previous IMU values.
"""

from __future__ import annotations

import torch

from mjlab_microduck.supermimic.context import ContextReader, xi_slices
from mjlab_microduck.supermimic.rotations import projected_gravity, quat_apply

OBS_DIM = 61
# uniform half-widths, env cfg: ang_vel 0.03, gravity 0.01, joint_pos 0.001, joint_vel 0.25
NOISE = (0.03, 0.01, 0.001, 0.25)


def imu_terms(s: torch.Tensor, xi: torch.Tensor, nq: int):
    q_imu = ContextReader.imu_quat(xi)
    w = quat_apply(q_imu, s[..., nq + 3 : nq + 6])
    g = quat_apply(q_imu, projected_gravity(s[..., 3:7]))
    return w, g


def actor_obs(
    s: torch.Tensor,
    qd_prev: torch.Tensor,
    last_action: torch.Tensor,
    commands: torch.Tensor,
    xi: torch.Tensor,
    q0: torch.Tensor,
    imu_prev: tuple[torch.Tensor, torch.Tensor] | None = None,
    imu_use_prev: torch.Tensor | None = None,
    noise: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Returns (obs (N,61), current imu (ang_vel, gravity) for the next step's delay).

    commands: (N,13) = [twist(3), head(4), body(6)].
    """
    nj = q0.shape[-1]
    nq = 7 + nj
    sl = xi_slices(nj)
    w, g = imu_terms(s, xi, nq)
    w_o, g_o = w, g
    if imu_prev is not None and imu_use_prev is not None:
        m = imu_use_prev[:, None].float()
        w_o = m * imu_prev[0] + (1 - m) * w
        g_o = m * imu_prev[1] + (1 - m) * g
    jp = s[..., 7:nq] + xi[..., sl["encoder_bias"]] - q0
    obs = torch.cat([w_o, g_o, jp, qd_prev, last_action, commands], dim=-1)
    if noise:
        half = torch.cat(
            [obs.new_full((3,), NOISE[0]), obs.new_full((3,), NOISE[1]), obs.new_full((nj,), NOISE[2]), obs.new_full((nj,), NOISE[3]), obs.new_zeros(nj + 13)]
        )
        obs = obs + (torch.rand_like(obs) * 2 - 1) * half
    return obs, (w, g)
