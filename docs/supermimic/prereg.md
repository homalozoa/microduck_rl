# Pre-registration: SuperMimic-style trainer vs PPO on Mjlab-Velocity-Flat-MicroDuck

Written and committed before any benchmark run. Smoke tests (64 envs, 5 iterations) and
unit tests are not benchmark runs. Any later change to this file is a separate commit and
is reported as a deviation in `results.md`.

## Question
Does the world-model + BPTT trainer (`train-supermimic`, this branch) reach a usable
Velocity-Flat gait in less wall-clock time than the repo's PPO config, both measured the
same way on the same GPU?

## Arms
- **PPO**: `uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
  --agent.save-interval 150 --agent.seed S --agent.logger tensorboard`, with the repo's
  `MicroduckRlCfg` otherwise unchanged. The save interval (about 158 s at 1.05 s per
  iteration) only adds checkpoints for the eval and does not change training.
- **SM**: `uv run train-supermimic --num-envs 4096 --seed S --max-hours 3`, with the
  `SMCfg` defaults frozen in the commit that carries this file: world model 3×2048,
  k = 8, UTD 0.4, H = 32, batch 4096, lr 2e-4, action noise 0.1 rad, replay 400 rows
  per env, torch.compile on. It uses the same env cfg (robot,
  BAM, domain randomization, pushes, observations, curricula).
- 3 seeds each: 0, 1, 2.

## Compute and timing
- All 6 runs go on linkff (1× RTX 4090, 48 GB), one at a time, with no other GPU job
  running. GPU type is therefore identical across arms.
- Budget: 3.0 h of training wall clock per run. A run that has not met the criterion by
  then is censored at 3.0 h.
- Wall clock is time since the training process started its first env step. It includes
  everything the trainer does: env stepping, world-model fitting, policy and critic
  updates, logging and checkpoint writes. It excludes process start-up and env
  construction (identical across arms) and the offline eval.
- Each checkpoint records its wall-clock time and env-step count: a sidecar JSON for SM,
  and for PPO the iteration number × 24 × 4096 plus a timestamp from the run log.
- Checkpoints are written about every 2.5 minutes: PPO every 150 iterations, SM every
  150 s of wall clock. Only checkpoints with wall clock ≤ 3.0 h count.
- Each run is followed by its eval, never concurrently with a training run.
- Measured before registration (64- and 4096-env smoke and timing runs): PPO about 94k
  env steps/s (1.05 s per iteration); SM about 6 env steps/s × 4096 = about 25k samples/s
  after a one-time compile of about 80 s.

## Episode-level eval (`scripts/supermimic/eval_velocity.py`)
- The policy is run deterministically on the training env cfg (`play=False`), so domain
  randomization, pushes, observation noise and delays are all on.
- 1024 envs × one full episode each (`episode_length_s` of the cfg), no auto-reset.
- Curricula are forced to their final stage: action-rate weight, standing fraction,
  head-pose ranges and CoM ranges. Every checkpoint is therefore judged on the final
  task, the same for both arms.
- Fixed eval seed 1234.
- Metrics, computed per episode and then averaged over episodes:
  - `fall_rate`: fraction of episodes ending in `fell_over` or `nan_state` before the
    time-out.
  - `lin_err`: mean over steps of ‖cmd_xy − v_xy,body‖ (m/s), on walking envs only
    (‖cmd‖ > 0.01), over the steps before any fall.
  - `yaw_err`: mean over steps of |cmd_ωz − ω_z| (rad/s), same masks.
  - `ret`: mean undiscounted episode return under the eval (final-stage) reward weights.

## Pilot disclosed
One 25-minute SM pilot (`pilot1`, seed 0) was run before registration to catch bugs. It
learned to stay up under the curriculum stage it was on. Training fall fraction was about
0.1 at 25 minutes, and the eval at its own stage gave fall_rate 0.107. Under the
final-stage eval it fell in 100% of episodes, within about 20 steps.

The env's curricula advance on env steps. PPO reaches the final stage (step 2000 × 24)
after about 35 minutes. At SM's measured rate, SM would reach it only after about
2.2 h. This is a property of running the unchanged env, and it is part of what the
benchmark measures. No SM default and no threshold was changed after the pilot. No PPO
pilot was run.

## Diagnostic (not used in the verdict)
The last checkpoint of every run is also evaluated with all curricula at their initial
stage (common_step_counter = 0). This separates "has not learned to walk" from "has not
yet been trained on the final-stage task".

## "Usable gait" criterion (primary, absolute)
AGENTS.md defines no numeric gait criterion, and the repo has no velocity eval script.
This one is fixed here:

**usable ⇔ fall_rate ≤ 0.05 AND lin_err ≤ 0.10 m/s AND yaw_err ≤ 0.25 rad/s**

The command ranges are |vx| ≤ 0.4, |vy| ≤ 0.3, |ωz| ≤ 1.0.

**Time-to-usable (T_use)** is the wall clock of the first checkpoint meeting the criterion
when the next checkpoint also meets it. This two-in-a-row rule guards against single-eval
luck. **Steps-to-usable (N_use)** is defined the same way in env steps.

## Secondary criterion (relative)
**usable-rel ⇔ fall_rate ≤ PPO_final_fall + 0.02 AND lin_err ≤ 1.10 × PPO_final_lin**

PPO_final is the 3-seed mean eval of each PPO run's last checkpoint. It is used if the
absolute criterion turns out to be unreachable for PPO in 3 h.

## Verdict rules (decided now)
- **Clear speedup**: all 3 SM seeds reach "usable", AND median T_use(SM) ≤ 0.67 ×
  median T_use(PPO), AND SM's final eval is not worse than PPO's: 3-seed mean fall_rate
  at most +0.02 and lin_err at most ×1.10.
- **Parity / no clear speedup**: both arms reach the criterion, but the speedup condition
  fails.
- **Negative**: fewer than 2 of 3 SM seeds reach the criterion within 3 h while PPO does,
  or SM's final eval is worse than PPO's beyond the margins above.
- If PPO itself does not reach the absolute criterion in 3 h, the same rules are applied
  with usable-rel. This is stated in the results.
- Only a "clear speedup" verdict leads to proposing SitStand next. Nothing further is run
  without the user asking.

## Also reported
- Each run's final training return, under each trainer's own logging. This is not
  comparable across arms: PPO logs its stochastic policy, SM logs its noisy-action policy.
- Env steps to criterion and env steps per second.
- Time split for SM: simulation, world-model fitting, policy update.
- World-model 1-step and 8-step error.
- Every penalty term is ≤ 0 in the env's episode reward logs (AGENTS.md sign check).

## Known confounds, stated up front
- The SM policy is deterministic with fixed exploration noise. PPO's is Gaussian with a
  learned std, and PPO's eval uses the mean action.
- The SM imagination uses analytic rewards that are tested against the env. Contact-gated
  terms (air_time, foot_swing_height, foot_slip, self_collisions) use the world model's
  predicted contacts, and their indicator functions give no gradient to the policy.
- Pushes, resets and command resamples are masked out of world-model windows.
