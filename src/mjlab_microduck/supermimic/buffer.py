"""Per-env ring replay buffer of real env steps (GPU).

Row (env e, slot t) holds everything at observation time t plus the action executed
from it:
  state s_t (S), actions a_t / a_{t-1} / a_{t-2} (3*NJ), xi (X), commands (13),
  contact flags (left, right, self) at t, air time (2), swing peak (2), head EMA (4),
  joint_vel seen by the actor at t (= qd_{t-1}, NJ), IMU (ang vel, gravity) of t-1 (6),
  next_ok: the transition t -> t+1 is clean physics (no reset, no push in between).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Layout:
    S: int
    NJ: int
    X: int

    def __post_init__(self):
        sizes = [
            ("state", self.S),
            ("acts", 3 * self.NJ),
            ("xi", self.X),
            ("cmd", 13),
            ("contact", 3),
            ("air", 2),
            ("peak", 2),
            ("ema", 4),
            ("qd_obs", self.NJ),
            ("imu_prev", 6),
        ]
        self.sl = {}
        o = 0
        for n, d in sizes:
            self.sl[n] = slice(o, o + d)
            o += d
        self.width = o


class ReplayBuffer:
    def __init__(self, num_envs: int, capacity: int, layout: Layout, device):
        self.n = num_envs
        self.cap = capacity
        self.L = layout
        self.data = torch.zeros(capacity, num_envs, layout.width, device=device)
        self.next_ok = torch.zeros(capacity, num_envs, dtype=torch.bool, device=device)
        self.ptr = 0
        self.count = 0
        self.device = device

    def add(self, row: torch.Tensor) -> int:
        """Write row (N, width) at the current slot; returns the slot index."""
        t = self.ptr
        self.data[t] = row
        self.next_ok[t] = False
        self.ptr = (self.ptr + 1) % self.cap
        self.count = min(self.count + 1, self.cap)
        return t

    def mark_next(self, slot: int, ok: torch.Tensor) -> None:
        self.next_ok[slot] = ok

    def field(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x[..., self.L.sl[name]]

    def sample_windows(self, batch: int, k: int):
        """k-step windows with every transition clean: (rows (k+1, B, W))."""
        assert self.count > k + 1
        tries = 0
        while True:
            # oldest valid start age = count-1, newest start whose window is complete = k (age)
            age = torch.randint(k + 1, self.count, (batch * 2,), device=self.device)
            start = (self.ptr - 1 - age) % self.cap
            env = torch.randint(0, self.n, (batch * 2,), device=self.device)
            idx = (start[None, :] + torch.arange(k + 1, device=self.device)[:, None]) % self.cap  # (k+1, 2B)
            ok = self.next_ok[idx[:-1], env[None, :]].all(0)
            sel = ok.nonzero().squeeze(-1)
            tries += 1
            if sel.numel() >= batch or tries > 20:
                sel = sel[:batch]
                if sel.numel() < batch:  # pad by repetition (very early in training only)
                    sel = sel[torch.randint(0, max(sel.numel(), 1), (batch,), device=self.device)]
                idx, env = idx[:, sel], env[sel]
                return self.data[idx, env[None, :]]

    def sample_starts(self, batch: int):
        """Start rows for imagination (any written row)."""
        age = torch.randint(0, self.count, (batch,), device=self.device)
        slot = (self.ptr - 1 - age) % self.cap
        env = torch.randint(0, self.n, (batch,), device=self.device)
        return self.data[slot, env]
