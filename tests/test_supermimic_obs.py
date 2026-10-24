"""SuperMimic trainer: the analytic 61-D actor obs equals the env's (GPU only).

Noise is switched off and the 0-1 step IMU delay pinned to 0 so the comparison is exact;
joint_vel keeps its fixed 1-step lag (reproduced from the previous state).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (mjlab / MuJoCo Warp)")


def test_actor_obs_matches_env():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.supermimic.context import ContextReader
    from mjlab_microduck.supermimic.kin import read_state
    from mjlab_microduck.supermimic.obs import actor_obs

    n = 128
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    cfg.scene.num_envs = n
    cfg.observations["actor"].enable_corruption = False
    for t in ("base_ang_vel", "projected_gravity"):
        cfg.observations["actor"].terms[t].delay_max_lag = 0
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    ctx = ContextReader(env)
    q0 = env.scene["robot"].data.default_joint_pos[0]
    obs, _ = env.reset()
    qd_prev = read_state(env)[:, -14:].clone()
    worst = torch.zeros(61, device="cuda:0")
    for _ in range(60):
        a = 0.3 * torch.randn(n, 14, device="cuda:0")
        s_before = read_state(env)
        obs, *_ , = env.step(a)
        s = read_state(env)
        cm = env.command_manager
        cmd = torch.cat([cm.get_command("twist"), cm.get_command("head_pose"), cm.get_command("body_pose")], -1)
        ana, _ = actor_obs(s, s_before[:, -14:], env.action_manager.action, cmd, ctx.read(), q0)
        ok = env.episode_length_buf > 1  # joint_vel delay buffer is reset with the episode
        if not ok.any():
            continue
        worst = torch.maximum(worst, (ana - obs["actor"])[ok].abs().max(0).values)
        qd_prev = s[:, -14:]
    names = ["ang_vel"] * 3 + ["gravity"] * 3 + ["joint_pos"] * 14 + ["joint_vel"] * 14 + ["actions"] * 14 + ["cmd"] * 13
    per = {}
    for nm, w in zip(names, worst.tolist()):
        per[nm] = max(per.get(nm, 0.0), w)
    print(per)
    assert all(v < 1e-4 for v in per.values()), per
