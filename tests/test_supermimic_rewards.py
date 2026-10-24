"""SuperMimic trainer: analytic Velocity-Flat rewards / kinematics vs the env (GPU only).

1. formula: every reward term recomputed from the env's own reward-time quantities
   matches ``RewardManager._step_reward`` (max abs error per term < 1e-4).
2. kinematics: FK + subtree angular momentum from the 41-d world-model state match the
   env's derived quantities at observation time (fresh forward()).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (mjlab / MuJoCo Warp)")

TASK = "Mjlab-Velocity-Flat-MicroDuck"
N = 256
STEPS = 120


@pytest.fixture(scope="module")
def rollout():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.supermimic.kin import env_kin
    from mjlab_microduck.supermimic.terms import Commands, Contacts, VelocityReward

    torch.manual_seed(0)
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = N
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.common_step_counter = 3000 * 24  # every curriculum at its final stage (all weights non-zero)
    vr = VelocityReward(env)
    rm = env.reward_manager
    contact = env.scene["feet_ground_contact"]
    selfc = env.scene["self_collision"]
    swing = rm.get_term_cfg("foot_swing_height").func
    names = list(rm.active_terms)
    rec = {"formula": []}
    orig = rm.compute

    def hooked(dt):
        d = contact.data
        h = env.scene["foot_height_scan"].data.heights
        in_air = (d.found == 0).float()
        peak = torch.where(in_air > 0, torch.maximum(swing.peak_heights, h), swing.peak_heights)
        c = Contacts(
            contact=(d.found > 0).float(),
            air_time=d.current_air_time.clone(),
            first_contact=contact.compute_first_contact(dt=env.step_dt).float(),
            peak_height=peak,
            self_collision=selfc.data.found.sum(-1).float(),
        )
        cm = env.command_manager
        cmd = Commands(cm.get_command("twist").clone(), cm.get_command("head_pose").clone(), cm.get_command("body_pose").clone())
        ema = getattr(env, "_head_bias_ema", None)
        jp = env.scene["robot"].data.joint_pos
        err = vr.head_error(jp, cmd.head)
        ema = torch.zeros_like(err) if ema is None else ema.clone()
        ema[env.episode_length_buf <= 1] = 0.0
        a = vr.head_alpha()
        ema = (1 - a) * ema + a * err
        k = env_kin(env)
        terms = vr.terms(k, c, cmd, env.action_manager.action.clone(), env.action_manager.prev_action.clone(), ema)
        out = orig(dt)
        w = vr.weights()
        ana = torch.stack([terms[n] * w[n] for n in names], -1)
        rec["formula"].append((ana, rm._step_reward.clone(), vr.total(terms), out.clone()))
        return out

    rm.compute = hooked
    env.reset()
    for _ in range(STEPS):
        a = 0.4 * torch.randn(N, 14, device="cuda:0")
        env.step(a)
    rm.compute = orig
    env.close()
    return names, rec


@pytest.fixture(scope="module")
def kin_rollout():
    """Separate env WITHOUT the reward hook: reading sensors inside the hook (before the env's
    own forward()/sense()) leaves the cached ray-sensor data one step stale at obs time."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.supermimic.context import ContextReader
    from mjlab_microduck.supermimic.kin import RobotKinematics, env_kin, read_state

    torch.manual_seed(1)
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = N
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.common_step_counter = 3000 * 24
    rk = RobotKinematics(env, "cuda:0")
    ctx = ContextReader(env)
    out = []
    lag_stats = []
    env.reset()
    for _ in range(STEPS):
        env.step(0.4 * torch.randn(N, 14, device="cuda:0"))
        k = rk.kin(read_state(env), ctx.read())
        r = env.scene["robot"]
        site_z = r.data.site_pos_w[:, r.find_sites(("left_foot", "right_foot"))[0], 2] - env.scene.env_origins[:, 2:3]
        sensor_h = env.scene["foot_height_scan"].data.heights.clone()
        ref = env_kin(env, foot_height=site_z.clamp_min(0.0))
        ref = type(ref)(**{f: getattr(ref, f).clone() for f in ref.__dataclass_fields__})
        lag_stats.append(((sensor_h - ref.foot_height).abs() > 1e-4).float().mean())
        out.append((k, ref, env.episode_length_buf.clone()))
    env.close()
    # The ray-cast foot_height_scan (what the env's foot_clearance / foot_swing_height read)
    # sometimes lags the true site height; FK is compared to the true site height instead.
    print(f"foot_height_scan differs from site z (>0.1 mm) on {100 * float(torch.stack(lag_stats).mean()):.1f}% of foot-steps")
    return out


def test_each_reward_term_matches_env(rollout):
    names, rec = rollout
    ana = torch.cat([a for a, _, _, _ in rec["formula"]])
    ref = torch.cat([r for _, r, _, _ in rec["formula"]])
    ok = torch.isfinite(ref).all(-1)
    errs = {}
    for i, n in enumerate(names):
        errs[n] = float((ana[ok, i] - ref[ok, i]).abs().max())
        print(f"{n:24s} max|err| {errs[n]:.2e}  mean|ref| {float(ref[ok, i].abs().mean()):.3e}")
    bad = {n: e for n, e in errs.items() if e > 1e-4}
    assert not bad, bad


def test_total_reward_matches_env(rollout):
    _, rec = rollout
    tot = torch.cat([t for _, _, t, _ in rec["formula"]])
    ref = torch.cat([r for _, _, _, r in rec["formula"]])
    assert float((tot - ref).abs().max()) < 1e-5


def test_state_kinematics_match_env(kin_rollout):
    """FK from the 41-d state vs the env at observation time.

    Foot height is compared with the true foot-site height (clamped at the ground like the
    ray sensor), not with the ray sensor, which lags by a step on some envs (printed).
    Envs in their first 2 steps after a reset are excluded (angmom sensor stale there).
    Reported as max and 99.9th percentile of the per-env max error; asserted on p99.9.
    """
    rec = {"kin": kin_rollout}
    fields = ("quat_w", "lin_vel_b", "ang_vel_b", "ang_vel_w", "joint_pos", "foot_height", "foot_vel_xy", "angmom")
    tol = {"quat_w": 1e-4, "lin_vel_b": 1e-3, "ang_vel_b": 1e-3, "ang_vel_w": 1e-3, "joint_pos": 1e-6, "foot_height": 1e-4, "foot_vel_xy": 1e-3, "angmom": 1e-4}
    bad = {}
    for f in fields:
        per = []
        for k, ref, eplen in rec["kin"]:
            a, b = getattr(k, f), getattr(ref, f)
            if f == "quat_w":
                a = a * torch.sign((a * b).sum(-1, keepdim=True))
            per.append((a - b).abs().reshape(a.shape[0], -1).max(-1).values[eplen > 2])
        per = torch.cat(per)
        if f in ("angmom", "foot_height"):
            for t, (k, ref, eplen) in enumerate(rec["kin"]):
                e = (getattr(k, f) - getattr(ref, f)).abs().reshape(eplen.shape[0], -1).max(-1).values
                e[eplen <= 2] = 0
                if float(e.max()) > tol[f]:
                    j = int(e.argmax())
                    print("  outlier", f, "step", t, "env", j, "eplen", int(eplen[j]), getattr(k, f)[j].tolist(), getattr(ref, f)[j].tolist())
        p999 = float(torch.quantile(per, 0.999))
        print(f"{f:12s} max|err| {float(per.max()):.2e}  p99.9 {p999:.2e}")
        if p999 > tol[f]:
            bad[f] = p999
    assert not bad, bad
