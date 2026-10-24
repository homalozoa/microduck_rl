"""SuperMimic-style trainer for Mjlab-Velocity-Flat-MicroDuck (paper Alg. A.1 structure).

Loop (per real env step, smduck.training.loop):
  (1) act in the real env with the deterministic policy + Gaussian exploration noise and
      record the step in the replay buffer;
  (2) at an update-to-data ratio `utd`: fit the world model on k-step replay windows;
  (3) improve the policy by BPTT through H imagined world-model steps (analytic
      observations / rewards / termination, terminal value from a critic fitted to GAE
      targets of the same rollout).

The env (robot, BAM actuators, DR, pushes, observations, curricula) is the unchanged
PPO task; only the learner differs. Imagination reads the live reward weights, so the
env's curricula apply to both the real and the imagined reward.
"""

from __future__ import annotations

import functools
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab_microduck.supermimic.buffer import Layout, ReplayBuffer
from mjlab_microduck.supermimic.context import ContextReader, XiNormalizer
from mjlab_microduck.supermimic.kin import RobotKinematics, read_state
from mjlab_microduck.supermimic.obs import OBS_DIM, actor_obs, imu_terms
from mjlab_microduck.supermimic.rotations import projected_gravity
from mjlab_microduck.supermimic.terms import Commands, Contacts, VelocityReward
from mjlab_microduck.supermimic.world_model import WorldModel, WorldModelCfg


@dataclass
class SMCfg:
    task: str = "Mjlab-Velocity-Flat-MicroDuck"
    num_envs: int = 4096
    seed: int = 0
    device: str = "cuda:0"
    log_dir: str = "logs/supermimic"
    run_name: str = "debug"
    max_hours: float = 3.0
    max_env_steps: int | None = None  # per env (iterations); for smoke tests
    # data / exploration
    action_noise_std: float = 0.1  # rad (action scale is 1.0)
    replay_capacity: int = 400  # rows per env
    warmup_steps: int = 40
    utd: float = 0.4  # updates per env step
    # world model
    wm: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_window: int = 8
    wm_batches_per_update: int = 10
    wm_batch: int = 2048
    wm_lr: float = 2e-4
    # actor / critic
    horizon: int = 32
    batch: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    critic_lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    hidden: tuple[int, ...] = (512, 256, 128)  # = PPO actor
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    obs_noise: bool = True
    fall_tau: float = 0.05  # rad, soft fell_over in imagination
    soft_tau_time: float = 0.02
    # bookkeeping
    log_every: int = 50
    ckpt_every_s: float = 150.0
    tf32: bool = True
    compile: bool = True  # torch.compile the imagination step and the world-model loss step


# ---------------------------------------------------------------------------- nets


class RunningNorm(nn.Module):
    """Obs normaliser updated from real env observations only (no gradient)."""

    def __init__(self, dim, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x):
        b = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + b
        d = m - self.mean
        self.mean += d * b / tot
        self.var = (self.var * self.count + v * b + d * d * self.count * b / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


def elu_mlp(i, hidden, o, last_gain=None):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    last = nn.Linear(d, o)
    if last_gain is not None:
        nn.init.orthogonal_(last.weight, gain=last_gain)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = RunningNorm(OBS_DIM)
        self.mlp = elu_mlp(OBS_DIM, hidden, 14, last_gain=0.01)

    def forward(self, obs):
        return self.mlp(self.norm(obs))


# ---------------------------------------------------------------------------- push / helpers


def _flag_push(func):
    @functools.wraps(func)
    def wrapped(env, env_ids, *args, **kwargs):
        m = getattr(env, "_sm_pushed", None)
        if m is not None:
            if env_ids is None:
                m[:] = True
            else:
                m[env_ids] = True
        return func(env, env_ids, *args, **kwargs)

    return wrapped


def make_env(task: str, num_envs: int, device: str, seed: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].func = _flag_push(cfg.events["push_robot"].func)
    env = ManagerBasedRlEnv(cfg, device=device)
    env._sm_pushed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    return env


# ---------------------------------------------------------------------------- trainer


class Trainer:
    def __init__(self, cfg: SMCfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        if cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.dev = cfg.device
        self.out = Path(cfg.log_dir) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.env = make_env(cfg.task, cfg.num_envs, cfg.device, cfg.seed)
        env = self.env
        self.robot = env.scene["robot"]
        self.nj = len(self.robot.joint_names)
        self.q0 = self.robot.data.default_joint_pos[0].clone()
        self.ctx = ContextReader(env)
        self.kin = RobotKinematics(env, cfg.device)
        self.rew = VelocityReward(env, soft_tau_time=cfg.soft_tau_time)
        self.contact = env.scene["feet_ground_contact"]
        self.selfc = env.scene["self_collision"]
        self.swing = env.reward_manager.get_term_cfg("foot_swing_height").func
        S = 2 * self.nj + 13
        self.L = Layout(S=S, NJ=self.nj, X=self.ctx.dim)
        self.buf = ReplayBuffer(cfg.num_envs, cfg.replay_capacity, self.L, cfg.device)
        self.wm = WorldModel(cfg.wm, self.nj, self.ctx.dim, cfg.device)
        self.policy = Policy(cfg.hidden).to(cfg.device)
        cdim = self.wm.spec.feat_dim + 13 + self.ctx.dim + 2 + 2 + 2 + 4 + self.nj
        self.critic = elu_mlp(cdim, cfg.critic_hidden, 1).to(cfg.device)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=cfg.wm_lr)
        self.pi_opt = torch.optim.Adam(self.policy.mlp.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.v_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.xi_norm: XiNormalizer | None = None
        self.cos_fall = math.cos(math.radians(70.0))
        self.fall_limit = math.radians(70.0)
        self._step_fn = torch.compile(self._imag_step, dynamic=False) if cfg.compile else self._imag_step
        if cfg.compile:
            self.wm.compile_loss()
        self.it = 0
        self.updates = 0
        self.wall = 0.0
        n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        print(f"[SM] params: world model {n(self.wm)/1e6:.2f}M policy {n(self.policy)/1e6:.3f}M critic {n(self.critic)/1e6:.3f}M; xi dim {self.ctx.dim}")

    # ------------------------------------------------------------------ real env I/O

    def _commands(self):
        cm = self.env.command_manager
        return torch.cat([cm.get_command("twist"), cm.get_command("head_pose"), cm.get_command("body_pose")], dim=-1)

    def _row(self, s, acts, xi, qd_obs, imu_prev):
        d = self.contact.data
        ema = getattr(self.env, "_head_bias_ema", None)
        ema = torch.zeros(self.env.num_envs, 4, device=self.dev) if ema is None else ema
        parts = [
            s,
            acts,
            xi,
            self._commands(),
            (d.found > 0).float(),
            (self.selfc.data.found.sum(-1, keepdim=True) > 0).float(),
            d.current_air_time,
            self.swing.peak_heights,
            ema,
            qd_obs,
            imu_prev,
        ]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ critic features

    def _critic_in(self, s, cmd, xi_n, contact, air, peak, ema, a_prev):
        return torch.cat([self.wm.spec.canonical(s), cmd, xi_n, contact, air * 5.0, peak * 20.0, ema, a_prev], dim=-1)

    # ------------------------------------------------------------------ world model update

    def wm_update(self):
        cfg = self.cfg
        L = self.L
        agg = {}
        for _ in range(cfg.wm_batches_per_update):
            rows = self.buf.sample_windows(cfg.wm_batch, cfg.wm_window)  # (k+1, B, W)
            states = rows[..., L.sl["state"]]
            acts = rows[:-1, :, L.sl["acts"]]
            xi_n = self.xi_norm(rows[0, :, L.sl["xi"]])
            labels = rows[1:, :, L.sl["contact"]]
            loss, m = self.wm.window_loss(states, acts, xi_n, labels)
            self.wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            m["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
            self.wm_opt.step()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v.detach()
        return {f"wm/{k}": float(v) / cfg.wm_batches_per_update for k, v in agg.items()}

    # ------------------------------------------------------------------ imagination

    def _imag_step(self, s, a1, a2, qd_obs, w_prev, g_prev, use_prev, contact, air, peak, ema, c, xi, xi_n, cmd, wvec):
        """One imagined step (compiled as a unit). Returns the next carry + reward, d, per-term means."""
        cfg = self.cfg
        nj, dt = self.nj, self.env.step_dt
        nq = 7 + nj
        cmds = Commands(cmd[:, :3], cmd[:, 3:7], cmd[:, 7:13])
        obs, (w_now, g_now) = actor_obs(s, qd_obs, a1, cmd, xi, self.q0, (w_prev, g_prev), use_prev, noise=cfg.obs_noise)
        a = self.policy(obs)
        s_next, logits = self.wm.step(s, a, a1, a2, xi_n)
        p = torch.sigmoid(logits)
        c_next = p[:, :2]
        air_next = (air + dt) * (1.0 - c_next)
        first = c_next * (1.0 - contact)
        k = self.kin.kin(s_next, xi)
        peak_upd = (1.0 - c_next) * torch.maximum(peak, k.foot_height) + c_next * peak
        alpha = self.rew.head_alpha()
        ema_next = (1 - alpha) * ema + alpha * self.rew.head_error(s_next[:, 7:nq], cmds.head)
        terms = self.rew.terms(k, Contacts(c_next, air_next, first, peak_upd, p[:, 2]), cmds, a, a1, ema_next, soft=True)
        tv = torch.stack(list(terms.values()), -1)  # (N, T)
        r = (tv * wvec).sum(-1) * dt
        g = projected_gravity(s_next[:, 3:7])
        tilt = torch.acos((-g[:, 2]).clamp(-1 + 1e-6, 1 - 1e-6))
        d = torch.sigmoid((tilt - self.fall_limit) / cfg.fall_tau)
        c_new = c * (1.0 - d)
        carry = (s_next, a, a1, s[:, nq + 6 :], w_now, g_now, use_prev, c_next, air_next, peak_upd * (1.0 - first), ema_next, c_new)
        x = self._critic_in(s_next, cmd, xi_n, c_next, air_next, peak_upd * (1.0 - first), ema_next, a)
        return carry, r, d, x, tv.detach().mean(0)

    def imagine(self):
        cfg = self.cfg
        L, nj = self.L, self.nj
        r0 = self.buf.sample_starts(cfg.batch)
        f = lambda n: r0[:, L.sl[n]]  # noqa: E731
        s = f("state")
        acts = f("acts")
        xi = f("xi")
        xi_n = self.xi_norm(xi)
        cmd = f("cmd")
        imu = f("imu_prev")
        c = torch.ones(cfg.batch, device=self.dev)
        carry = (s, acts[:, nj : 2 * nj], acts[:, 2 * nj :], f("qd_obs"), imu[:, :3], imu[:, 3:], (torch.rand(cfg.batch, device=self.dev) < 0.5).float(),
                 f("contact")[:, :2], f("air"), f("peak"), f("ema"), c)
        w = self.rew.weights()
        wvec = torch.tensor([w[n] if w[n] != 0.0 else 0.0 for n in self.rew.names], device=self.dev)
        rewards, dhats, survs = [], [], [c]
        crit_in = [self._critic_in(s, cmd, xi_n, carry[7], carry[8], carry[9], carry[10], carry[1])]
        tsum = 0.0
        for _ in range(cfg.horizon):
            carry, r, d, x, tm = self._step_fn(*carry, xi, xi_n, cmd, wvec)
            rewards.append(r)
            dhats.append(d)
            survs.append(carry[-1])
            crit_in.append(x)
            tsum = tsum + tm
        tsum = (tsum / cfg.horizon).tolist()
        return torch.stack(rewards), torch.stack(dhats), torch.stack(survs), torch.stack(crit_in), dict(zip(self.rew.names, tsum))

    def pi_update(self):
        cfg = self.cfg
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        R, D, C, X, terms = self.imagine()
        H = R.shape[0]
        v_H = self.critic(X[-1]).squeeze(-1)
        disc = cfg.gamma ** torch.arange(H, device=self.dev, dtype=R.dtype)
        J = ((disc[:, None] * C[:-1] * R).sum(0) + cfg.gamma**H * C[-1] * v_H).mean()
        loss = -J / H
        self.pi_opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.policy.mlp.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.mlp.parameters(), cfg.grad_clip_value)
        self.pi_opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        # critic on GAE targets of this rollout
        with torch.no_grad():
            Xd = X.detach()
            H1, n, dx = Xd.shape
            V = self.critic(Xd.reshape(-1, dx)).reshape(H1, n)
            Rd, Dd = R.detach(), D.detach()
            Rt = V[H].clone()
            tgt = torch.empty_like(Rd)
            for t in range(H - 1, -1, -1):
                Rt = Rd[t] + cfg.gamma * (1.0 - Dd[t]) * ((1 - cfg.gae_lambda) * V[t + 1] + cfg.gae_lambda * Rt)
                tgt[t] = Rt
            Xf, Y = Xd[:-1].reshape(-1, dx), tgt.reshape(-1)
        N = Y.shape[0]
        mb = N // cfg.critic_minibatches
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(N, device=self.dev)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                closs = F.mse_loss(self.critic(Xf[idx]).squeeze(-1), Y[idx])
                self.v_opt.zero_grad(set_to_none=True)
                closs.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.v_opt.step()
        m = {
            "pi/imag_return": float((C[:-1] * R).sum(0).mean()),
            "pi/imag_survival_H": float(C[-1].mean()),
            "pi/grad_norm": float(gn),
            "pi/critic_loss": float(closs.detach()),
            "pi/value_mean": float(V.mean()),
        }
        m.update({f"imag_r/{k}": v for k, v in terms.items()})
        return m

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str, final: bool = False):
        meta = {"wall_s": self.wall, "iteration": self.it, "env_steps": self.it * self.cfg.num_envs, "updates": self.updates}
        ck = {"policy": self.policy.state_dict(), "meta": meta}
        if final:
            ck.update(wm=self.wm.state_dict(), critic=self.critic.state_dict(), xi_norm=self.xi_norm.state_dict() if self.xi_norm else None)
        torch.save(ck, self.out / f"{name}.pt")
        (self.out / f"{name}.json").write_text(json.dumps(meta))

    # ------------------------------------------------------------------ main loop

    def run(self):
        cfg = self.cfg
        env = self.env
        nj = self.nj
        obs, _ = env.reset()
        obs = obs["actor"]
        zeros = torch.zeros(env.num_envs, nj, device=self.dev)
        a_hist = [zeros, zeros]  # a_{t-1}, a_{t-2}
        qd_prev = read_state(env)[:, -nj:]
        imu_prev = torch.zeros(env.num_envs, 6, device=self.dev)
        ep_ret = torch.zeros(env.num_envs, device=self.dev)
        ep_len = torch.zeros(env.num_envs, device=self.dev)
        fin_ret, fin_len, fin_fall = [], [], []
        agg, agg_n = {}, 0
        tm = {"sim": 0.0, "wm": 0.0, "pi": 0.0}
        acc = 0.0
        t_start = time.time()
        last_ck = -1e9
        self.save("ckpt_0000000")
        while True:
            self.wall = time.time() - t_start
            if self.wall / 3600 > cfg.max_hours or (cfg.max_env_steps is not None and self.it >= cfg.max_env_steps):
                break
            if self.wall - last_ck >= cfg.ckpt_every_s:
                self.save(f"ckpt_{self.it:07d}")
                last_ck = self.wall
            # ---- (1) real env step
            t0 = time.perf_counter()
            s = read_state(env)
            xi = self.ctx.read()
            if self.xi_norm is None:
                self.xi_norm = XiNormalizer(xi)
            with torch.no_grad():
                self.policy.norm.update(obs)
                a = self.policy(obs) + cfg.action_noise_std * torch.randn(env.num_envs, nj, device=self.dev)
            row = self._row(s, torch.cat([a, a_hist[0], a_hist[1]], -1), xi, qd_prev, imu_prev)
            slot = self.buf.add(row)
            env._sm_pushed[:] = False
            obs_d, rew, term, tout, _ = env.step(a)
            obs = obs_d["actor"]
            done = term | tout
            self.buf.mark_next(slot, ~done & ~env._sm_pushed)
            # IMU of this (pre-step) state for next row's delay model
            w_imu, g_imu = imu_terms(s, xi, 7 + nj)
            imu_prev = torch.cat([w_imu, g_imu], -1)
            qd_prev = s[:, -nj:]
            a_hist = [a, a_hist[0]]
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for h in a_hist:
                    h[ids] = 0.0
                qd_prev = qd_prev.clone()
                qd_prev[ids] = 0.0
                imu_prev[ids] = 0.0
            ep_ret += rew
            ep_len += 1
            if done.any():
                ids = done.nonzero().squeeze(-1)
                fin_ret += ep_ret[ids].tolist()
                fin_len += ep_len[ids].tolist()
                fin_fall += term[ids].float().tolist()
                ep_ret[ids] = 0
                ep_len[ids] = 0
            torch.cuda.synchronize()
            tm["sim"] += time.perf_counter() - t0
            self.it += 1
            # ---- (2)+(3) updates
            if self.it >= cfg.warmup_steps:
                acc += cfg.utd
                while acc >= 1.0:
                    acc -= 1.0
                    t0 = time.perf_counter()
                    m = self.wm_update()
                    torch.cuda.synchronize()
                    tm["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    m.update(self.pi_update())
                    torch.cuda.synchronize()
                    tm["pi"] += time.perf_counter() - t0
                    self.updates += 1
                    for k, v in m.items():
                        agg[k] = agg.get(k, 0.0) + v
                    agg_n += 1
            if self.it % cfg.log_every == 0:
                ret = sum(fin_ret) / len(fin_ret) if fin_ret else float("nan")
                eln = sum(fin_len) / len(fin_len) if fin_len else float("nan")
                fall = sum(fin_fall) / len(fin_fall) if fin_fall else float("nan")
                rec = {"it": self.it, "wall_h": self.wall / 3600, "env_steps": self.it * env.num_envs, "updates": self.updates, "train/ep_return": ret, "train/ep_len": eln, "train/fall_frac": fall, "train/n_eps": len(fin_ret)}
                rec.update({k: v / max(agg_n, 1) for k, v in agg.items()})
                rec.update({f"time/{k}_s": v for k, v in tm.items()})
                with open(self.out / "log.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(
                    f"it {self.it} | {self.wall/3600:.3f} h | upd {self.updates} | ret {ret:.2f} len {eln:.0f} fall {fall:.2f} (n={len(fin_ret)}) | "
                    f"wm nse1 {rec.get('wm/nse_1step', float('nan')):.4f} nse8 {rec.get('wm/nse_kstep', float('nan')):.4f} cacc {rec.get('wm/contact_acc', float('nan')):.3f} | "
                    f"imag ret {rec.get('pi/imag_return', float('nan')):.3f} surv {rec.get('pi/imag_survival_H', float('nan')):.3f} gn {rec.get('pi/grad_norm', float('nan')):.2f} | "
                    f"t sim/wm/pi {tm['sim']:.0f}/{tm['wm']:.0f}/{tm['pi']:.0f} s",
                    flush=True,
                )
                fin_ret, fin_len, fin_fall = [], [], []
                agg, agg_n = {}, 0
        self.save(f"ckpt_{self.it:07d}")
        self.save("final", final=True)
        env.close()
