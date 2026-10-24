"""Print the Velocity-Flat env layout the SuperMimic trainer depends on."""
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa: F401

cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
cfg.scene.num_envs = 8
env = ManagerBasedRlEnv(cfg, device="cuda:0")
r = env.scene["robot"]
print("step_dt", env.step_dt, "physics dt", env.sim.mj_model.opt.timestep, "decimation", cfg.decimation, "ep len s", cfg.episode_length_s)
print("joints", r.joint_names)
print("nq nv", env.sim.mj_model.nq, env.sim.mj_model.nv, "bodies", r.body_names)
print("sites", r.site_names)
print("default q", r.data.default_joint_pos[0].tolist())
om = env.observation_manager
for g in om.active_terms:
    print("group", g, om.group_obs_dim[g], list(zip(om.active_terms[g], om.group_obs_term_dim[g])))
    print("  cfg history", getattr(cfg.observations[g], "history_length", None), "concat", getattr(cfg.observations[g], "concatenate_terms", None))
rm = env.reward_manager
for n in rm.active_terms:
    print("reward", n, rm.get_term_cfg(n).weight)
print("terminations", env.termination_manager.active_terms)
print("events", {m: env.event_manager.active_terms[m] for m in env.event_manager.active_terms})
print("sensors", list(env.scene.sensors.keys()))
obs, _ = env.reset()
for k in range(3):
    obs, rew, term, tout, ex = env.step(torch.zeros(8, 14, device="cuda:0"))
print("step reward terms", {k: v[:2].tolist() for k, v in rm._step_reward.items()} if hasattr(rm, "_step_reward") else dir(rm))
fs = env.scene["feet_ground_contact"].data
print("contact", {k: (getattr(fs, k).shape if getattr(fs, k) is not None else None) for k in ("found", "force", "current_air_time", "last_air_time", "current_contact_time")})
hs = env.scene["foot_height_scan"].data
print("heights", hs.heights[:2], "site z", r.data.site_pos_w[:2, :, 2])
print("expanded fields", env.sim.expanded_fields if hasattr(env.sim, "expanded_fields") else "?")
m = env.sim.model
for f in ("body_ipos", "body_mass", "dof_armature", "geom_friction", "body_inertia"):
    print(f, tuple(getattr(m, f).shape))
for a in r.actuators:
    print(type(a).__name__, a.vin_tensor.shape, a.vin_drop_gain.shape if a.vin_drop_gain is not None else None, a.friction_scale.shape, [x for x in dir(a) if "lag" in x or "delay" in x])
print("encoder bias", [x for x in dir(r.data) if "bias" in x])
print("imu", getattr(env, "_imu_misalign_quat", None) is not None)
