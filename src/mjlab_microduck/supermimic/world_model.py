"""World model: state spec + dynamics MLP + contact heads (after smduck/world_model).

State integration, canonical (heading-invariant) features and the normalised error are
smduck's StateSpec with the Microduck scales (K_LEN = 0.2/0.78 size rule). Changes for the
velocity task:
  * inputs: canonical state, a_t, a_{t-1}, a_{t-2} (BAM command delay of 3-6 physics
    steps spans up to 1.5 control steps), normalised DR context xi;
  * extra outputs: logits of foot contact (left, right) and of a self-collision at t+1,
    trained with BCE on the env's contact sensors (the contact-gated reward terms read them);
  * no learned termination head: Velocity-Flat's fell_over is an analytic tilt test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab_microduck.supermimic.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

K_LEN = 0.2 / 0.78
K_VEL = math.sqrt(K_LEN)
K_ANG = 1.0 / math.sqrt(K_LEN)
N_LOGITS = 3  # contact left, contact right, self-collision


def _conj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    dt: float = 0.02
    err_scale: tuple[float, ...] = (0.05 * K_LEN, 0.1, 0.5 * K_VEL, 1.0 * K_ANG, 0.1, 2.0 * K_ANG)
    out_scale: tuple[float, ...] = (0.3 * K_VEL, 0.5 * K_ANG, 2.0 * K_ANG, 0.01 * K_LEN, 0.02, 0.02)
    in_scale: tuple[float, ...] = (1.0 * K_LEN, 1.0, 1.0 * K_VEL, 2.0 * K_ANG, 1.0, 5.0 * K_ANG)
    contact_loss_weight: float = 1.0


class StateSpec:
    def __init__(self, cfg: WorldModelCfg, nj: int, device):
        self.nj = nj
        self.nq = 7 + nj
        self.dt = cfg.dt
        rep = lambda sc, sz: torch.cat([torch.full((n,), float(s)) for s, n in zip(sc, sz)]).to(device)  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, nj, nj))
        self.out_scale = rep(cfg.out_scale, (3, 3, nj, 3, 3, nj))
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, nj, nj))
        self.feat_dim = 1 + 6 + 3 + 3 + 2 * nj
        self.delta_dim = 12 + 2 * nj  # dv_h, dw_b, dqd, dp_h, dth_b, dq

    def split(self, s):
        nj, j1 = self.nj, self.nq
        return s[..., 0:3], s[..., 3:7], s[..., 7:j1], s[..., j1 : j1 + 3], s[..., j1 + 3 : j1 + 6], s[..., j1 + 6 : j1 + 6 + nj]

    def canonical(self, s):
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(_conj(hq), q))
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), quat_apply_inverse(hq, v), w, qj, qd], dim=-1)
        return feat / self.in_scale

    def integrate(self, s, delta):
        d = delta * self.out_scale
        nj = self.nj
        o = 6 + nj
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6:o]
        dp_h, dth_b, dq = d[..., o : o + 3], d[..., o + 3 : o + 6], d[..., o + 6 : o + 6 + nj]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        p_new = p + self.dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(self.dt * w_new + dth_b)))
        qj_new = qj + self.dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    def error_vector(self, s_pred, s_true):
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        rot = rotvec_from_quat(quat_mul(_conj(quat_normalize(q2)), quat_normalize(q1)))
        return torch.cat([p1 - p2, rot, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1) / self.err_scale


def mlp(i, hidden, o, act=nn.SiLU):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, nj: int, xi_dim: int, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg, nj, device)
        self.nj = nj
        self.net = mlp(self.spec.feat_dim + 3 * nj + xi_dim, cfg.hidden, self.spec.delta_dim + N_LOGITS)
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)  # identity (constant-velocity) dynamics, p(contact) = 0.5 at init
        self.to(device)

    def step(self, s, a, a1, a2, xi_n):
        """-> (s_{t+1}, logits (…,3)) for actions a_t, a_{t-1}, a_{t-2}."""
        out = self.net(torch.cat([self.spec.canonical(s), a, a1, a2, xi_n], dim=-1))
        d = self.spec.delta_dim
        return self.spec.integrate(s, out[..., :d]), out[..., d:]

    def _loss_step(self, s, a, a1, a2, xi_n, s_true, label):
        s_next, logits = self.step(s, a, a1, a2, xi_n)
        e = self.spec.error_vector(s_next, s_true)
        bce = F.binary_cross_entropy_with_logits(logits, label)
        acc = ((logits[:, :2] > 0).float() == label[:, :2]).float().mean()
        return s_next, (e * e).sum(-1).mean(), bce, (e * e).mean().detach(), acc.detach()

    def compile_loss(self):
        self._loss_step_fn = torch.compile(self._loss_step, dynamic=False)

    def window_loss(self, states, acts, xi_n, labels):
        """states (k+1,n,S), acts (k+2... see buffer) -> loss, metrics.

        acts: (k, n, 3*nj) = [a_t, a_{t-1}, a_{t-2}] per transition; labels (k, n, 3) of s_{t+1}.
        """
        k = acts.shape[0]
        nj = self.nj
        s = states[0]
        st = 0.0
        ct = 0.0
        nse = []
        acc = 0.0
        fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            a = acts[j]
            s, se, bce, n_, ac = fn(s, a[:, :nj].contiguous(), a[:, nj : 2 * nj].contiguous(), a[:, 2 * nj :].contiguous(), xi_n, states[j + 1].contiguous(), labels[j].contiguous())
            st = st + se
            ct = ct + bce
            nse.append(n_)
            acc = acc + ac
        loss = st / k + self.cfg.contact_loss_weight * ct / k
        m = {"state_loss": (st / k).detach(), "contact_bce": (ct / k).detach(), "nse_1step": nse[0], "nse_kstep": nse[-1], "contact_acc": acc / k}
        return loss, m
