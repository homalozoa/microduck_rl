"""SuperMimic trainer: CPU unit tests (no simulator)."""

import torch

from mjlab_microduck.supermimic.buffer import Layout, ReplayBuffer
from mjlab_microduck.supermimic.world_model import WorldModel, WorldModelCfg

NJ = 14
S = 2 * NJ + 13


def _state(n):
    s = torch.zeros(n, S)
    s[:, 2] = 0.12
    s[:, 3] = 1.0
    s[:, 7 + NJ : 7 + NJ + 3] = torch.randn(n, 3) * 0.1
    s[:, -NJ:] = torch.randn(n, NJ)
    return s


def test_world_model_starts_as_constant_velocity():
    wm = WorldModel(WorldModelCfg(hidden=(32, 32)), NJ, xi_dim=5, device="cpu")
    s = _state(8)
    s2, logits = wm.step(s, torch.zeros(8, NJ), torch.zeros(8, NJ), torch.zeros(8, NJ), torch.zeros(8, 5))
    dt = wm.cfg.dt
    assert torch.allclose(s2[:, 0:3], s[:, 0:3] + dt * s[:, 7 + NJ : 10 + NJ], atol=1e-6)
    assert torch.allclose(s2[:, 7 : 7 + NJ], s[:, 7 : 7 + NJ] + dt * s[:, -NJ:], atol=1e-6)
    assert torch.allclose(logits, torch.zeros_like(logits))


def test_world_model_window_loss_is_differentiable():
    wm = WorldModel(WorldModelCfg(hidden=(32, 32)), NJ, xi_dim=5, device="cpu")
    k, n = 4, 16
    states = torch.stack([_state(n) for _ in range(k + 1)])
    loss, m = wm.window_loss(states, torch.randn(k, n, 3 * NJ), torch.randn(n, 5), torch.randint(0, 2, (k, n, 3)).float())
    loss.backward()
    assert torch.isfinite(loss)
    assert wm.net[-1].weight.grad.abs().sum() > 0
    assert m["nse_kstep"] >= 0


def test_buffer_windows_never_cross_resets_or_pushes():
    L = Layout(S=4, NJ=1, X=1)
    buf = ReplayBuffer(num_envs=3, capacity=50, layout=L, device="cpu")
    for t in range(60):
        row = torch.full((3, L.width), float(t))
        row[:, 1] = torch.arange(3).float()  # env id
        slot = buf.add(row)
        ok = torch.tensor([True, t % 7 != 0, False])  # env 2 never clean
        buf.mark_next(slot, ok)
    rows = buf.sample_windows(200, 4)
    stamps = rows[..., 0]  # (k+1, B)
    assert torch.all(stamps[1:] - stamps[:-1] == 1)  # consecutive real steps, no wrap
    env = rows[0, :, 1]
    assert not torch.any(env == 2)
    e1 = env == 1
    assert not torch.any(stamps[:-1, e1] % 7 == 0)
