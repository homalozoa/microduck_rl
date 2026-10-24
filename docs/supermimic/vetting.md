# Vetting: world-model / first-order BPTT vs PPO for reward-based biped locomotion

Date: 2026-10-24. Scope: is a SuperMimic-style trainer (learned world model, policy
trained by backprop through imagined rollouts) worth building instead of / alongside
PPO for flat-ground **velocity tracking** on the ~25 cm 14-DOF microduck, with a learned
voltage-servo actuator model, backlash and domain randomization (DR)?

Rules followed: only sources actually opened are cited; numbers are quoted as read on
the page. Where a page did not give a number, that is said.

## Note on "SuperMimic"

The vetting subagent's web search found nothing under that name. That does not mean the work
is absent. It is arXiv 2610.28480, "Scaling Humanoid Whole-Body Control with One
Differentiable World Model" (CoRL 2026), as cited in the README of the local reproduction
`/Users/homalozoax/supermimic-repro` (opened). There is also a public reimplementation,
CuboiLeo/SuperMimic-Reproduced @d9be385 (MIT; its licence is copied into
`src/mjlab_microduck/supermimic/`). No claims are taken from the paper text here. The
evidence for SuperMimic on this project is the user's own measurements (BFM project, given
in this task's brief and not re-verified here):
G1 motion tracking reproduced at 0.862, wall-clock win over PPO on G1 (0.82 in ~2.96 h vs
PPO best 0.44 in 3 h), a confounded microduck motion-tracking result, and world-model error
15-30x larger in contact-heavy phases. All of these are **motion imitation**, not
reward-based velocity tracking, so they do not settle the question below.

## Sources opened

| # | Source | URL |
|---|--------|-----|
| S1 | Xu et al., Accelerated Policy Learning with Parallel Differentiable Simulation (SHAC), ICLR 2022 | https://ar5iv.labs.arxiv.org/html/2204.07137 |
| S2 | Georgiev et al., Adaptive Horizon Actor-Critic (AHAC), ICML 2024 (abstract) | https://arxiv.org/abs/2405.17784 |
| S3 | Schwarke et al., Learning Quadrupedal Locomotion via Differentiable Simulation / Learning Deployable Locomotion Control via Differentiable Simulation (ETH) | https://arxiv.org/html/2404.02887v2 |
| S4 | Song, Kim, Scaramuzza, Learning Quadruped Locomotion Using Differentiable Simulation, CoRL 2024 | https://arxiv.org/html/2403.14864v4 |
| S5 | Amigo et al., First Order Model-Based RL through Decoupled Backpropagation (DMO), CoRL 2025 (abstract page) | https://proceedings.mlr.press/v305/amigo25a.html |
| S6 | Same paper, full text | https://arxiv.org/html/2509.00215v2 |
| S7 | Robotic World Model (RWM): A Neural Network Simulator for Robust Policy Optimization in Robotics | https://arxiv.org/html/2501.10100 |
| S8 | Suh, Simchowitz, Zhang, Tedrake, Do Differentiable Simulators Give Better Policy Gradients?, ICML 2022 (abstract) | https://arxiv.org/abs/2202.00817 |
| S9 | Onoda et al., Does "Do Differentiable Simulators Give Better Policy Gradients?" Give Better Policy Gradients? (2026, abstract) | https://arxiv.org/abs/2604.18161v1 |
| S10 | Wu et al., DayDreamer: World Models for Physical Robot Learning (project page) | http://danijar.com/project/daydreamer/ |
| S11 | MuscleMimic (arXiv 2603.25544) - opened while searching for SuperMimic; motion imitation, no PPO comparison; not relevant evidence | https://arxiv.org/abs/2603.25544 |

## Criterion 1 - someone did it with a large gain vs PPO

| Work | Exact claim (as read) | Setting | Relevance to us |
|------|----------------------|---------|-----------------|
| S1 SHAC | Humanoid MTU (152 actuators): "over 17x faster than PPO" wall clock, "382x more sample-efficient than PPO"; matched PPO reward in "about 35 minutes and 1.7M simulation steps". Ant: SHAC "less than 15 minutes" vs SE-MBPO "88 hours". | True differentiable sim (dFlex), forward-running reward, no DR, no sim-to-real, no actuator model. | Largest gains are on very high-DOF muscle models; a 14-DOF robot is far from that regime. Reward is "run forward", not command tracking. |
| S2 AHAC | "attaining 40% more reward across a set of locomotion tasks" vs MFRL baselines, "improved wall-clock-time efficiency". | Differentiable sim, contact-rich locomotion benchmarks. | A reward gain, not a time-to-solve gain; still true-sim gradients. |
| S3 ETH (SHAC on legged) | SHAC 2048 samples/iter vs PPO 49152; "over an order of magnitude" more sample-efficient; "nearly identical final rewards". Per-iteration wall clock SHAC 0.330 s vs PPO 0.183 s. ANYmal D velocity tracking deployed zero-shot on hardware. | Warp differentiable sim, PD actuator fitted to real data, DR on friction [0.5,1.25], base mass +-5 kg, pushes. | Closest to our task (velocity tracking + DR + real robot). But: gain is **sample** efficiency; **wall clock per iteration is ~1.8x slower** than PPO, and no overall wall-clock win is claimed. |
| S4 Song et al. | "requires only roughly 64 robots and 200 training iterations ... while PPO still performs less after 1000 training iterations with 1024 robots". Total training time per 1000 iterations essentially equal (e.g. 1024 robots: PPO 11.44 min vs ours 11.55 min). Mini Cheetah zero-shot. | Simplified single-rigid-body surrogate for gradients, IsaacGym for forward; Raibert foot placement and gait schedule supplied. | Gain is with few envs; at 1024 envs final rewards converge (-2.80 vs -2.69). Needs hand-designed gait priors; no DR experiments. |
| S6 DMO (learned model, decoupled) | "fewer than 4 million samples, over ten times less than PPO" (PPO run 160M); wall clock "up to 20% improvement". Go2 velocity tracking deployed on hardware; Go2 bipedal balancing deployed. | Real sim forward pass, gradients through a **learned MLP model** - the nearest published analogue of a world-model BPTT trainer on legged robots. DR only on joint friction. | Strongest direct evidence. Note the honest size: ~10x samples but only **up to 20% wall clock**. |
| S10 DayDreamer | A quadruped learned to "roll off its back, stand up, and walk from scratch and without resets in only 1 hour". | On the real robot, no simulator. | Shows world models shine when samples are expensive (real world). Not our regime: we have a cheap GPU sim. Page gives no PPO/SAC comparison. |

Takeaway: large **sample-efficiency** gains (10x-382x) are well documented. Large
**wall-clock** gains vs a GPU-parallel PPO on a low-DOF legged velocity task are not: S3
is slower per iteration, S4 equal time per iteration, S6 up to 20%.

## Criterion 2 - nobody did exactly this; first-principles view

Nobody (in sources opened) trained a ~25 cm 14-DOF biped velocity tracker with a learned
voltage-servo actuator model + backlash + broad DR via world-model BPTT. Reasoning:

1. **What we would gain is samples, and samples are not our bottleneck.** PPO at 4096
   envs already finishes in ~1-2 h; the sim is cheap. S3/S4/S6 all show the first-order
   win shrinks to roughly parity or +20% in wall clock once envs are many. Expected
   headroom: maybe 1.2-2x wall clock in a good case, not 10x.
2. **Backlash is a discontinuity by construction.** A dead-band in the gear train is a
   piecewise-flat map: gradient zero inside the band, a jump at engagement. That is
   exactly the "stiffness or discontinuities" S8 says compromise first-order estimates,
   stacked on top of foot contact. A learned model smooths it (helping gradients) but
   then the gradient is of a *fictional* smooth actuator - bias, not variance.
3. **Learned voltage-servo actuator model helps a bit.** It is a smooth neural net, so
   differentiable; it can be dropped straight into the world model. But it is
   history-dependent (latency, ~40 ms extra delay found earlier on the KPower robot),
   which lengthens the effective BPTT path and raises exploding-gradient risk (S1/S3).
4. **DR fights the world model.** Each DR sample (friction, mass, latency, backlash
   width) changes the dynamics. The model must either be conditioned on DR parameters
   (privileged input, as S6 does for base velocity / foot forces) or average over them,
   which smooths away exactly the variation DR is meant to expose. PPO needs nothing
   extra here.
5. **Small robot = light links, high relative stiffness.** S3 found gradient norm stays
   "well below 1.0" for a 6.2 kg quadruped but "grows exponentially" for ANYmal and
   "blocks learning beyond 16 steps". Which side a 25 cm biped falls on is unknown; a
   biped's balance is closer to unstable (chaotic) than a quadruped's, which argues for
   short horizons (~12-32 steps) and a critic, i.e. SHAC/AHAC structure, not long BPTT.
6. **Rewards must be differentiable** (S4, S6): termination penalties, alive bonuses and
   discrete gait/contact terms give zero gradient. Our PPO reward set would need redesign.

## Criterion 3 - shown not to work / known failure modes

- **Plain long-horizon BPTT explodes.** S1: Humanoid at H = 1000 gives gradient norm
  "exceeds 10^6"; best SHAC horizon on Ant was h = 16 or 32. S6: "DMO-BPTT did not
  perform well" on DFlex.
- **Exponential gradient growth on a real legged robot model.** S3, ANYmal: blocks
  learning beyond 16 steps; they ran horizon 12.
- **Soft contact gives gradients but bad policies.** S3: soft-contact policy return
  2231 in training falls to 325 under hard-contact evaluation (episode 9.69 s -> 1.58 s).
- **Contact discontinuity bias.** S8 (stiffness/discontinuities compromise first-order
  estimator); S2 (FO-MBRL "susceptible to sampling error in scenarios involving stiff
  dynamics"); S9 ("discontinuous dynamics cause bias"), though S9 concludes "variance
  control often matters more in practical deployments".
- **First-order through a learned model fails on legged locomotion in one direct test.**
  S7 (RWM): SHAC through the world model "fails to converge", produced "chaotic robot
  behaviors during training", and "SHAC and Dreamer fail to produce a deployable policy";
  their working recipe was PPO on imagined rollouts (MBPO-PPO).
- **Model exploitation.** S7: "Model inaccuracies can be exploited during policy
  learning"; SHAC "struggles with high and fluctuating model error".
- **Exploration.** S4 needed Raibert foot placement because velocity loss alone "cannot
  explore foot placement"; S6: DMO-SHAC and DMO-BPTT "explored poorly on the bipedal task".
- **Coupled forward pass through the model is worse.** S6 ablation: decoupled (real sim
  forward, model only for gradients) "achieves nearly double the asymptotic performance"
  vs model-based forward - i.e. rolling out *in* the world model (Dreamer/SuperTrack
  style) is the weaker variant.

## Verdict

**GO-with-caveats - as a bounded experiment, not a replacement for PPO.** Leaning toward
AVOID if the goal is "faster training of the production policy".

Honest headroom: published evidence supports ~10x fewer samples, but only parity to ~20%
wall-clock gain at large env counts (S3, S4, S6). Since PPO already converges in ~1-2 h at
4096 envs on a cheap sim, the realistic best case is saving tens of minutes per run, at the
cost of a new trainer, differentiable reward redesign, and new failure modes (backlash and
contact bias, exploding gradients, model exploitation, poor exploration on bipeds). The one
direct legged world-model test of first-order gradients (S7) failed; the one success (S6)
used the real simulator for the forward pass.

If pursued, the caveats are:
1. Use the DMO recipe (S6): roll out in the real MuJoCo/GPU sim, use the learned model
   only for gradients; do not roll out in imagination.
2. SHAC/AHAC structure: short horizon (start 12-32), value critic, gradient-norm logging
   per horizon; truncate at stiff contact (AHAC idea).
3. Condition the model on DR parameters (privileged inputs); smooth backlash only in the
   gradient path, keep the true dead-band in the forward sim.
4. Keep PPO as baseline and as fallback; success criterion = equal velocity-tracking
   error on the *same* DR eval in < 50% of PPO wall clock. Kill the experiment if not met
   within a fixed budget (e.g. 1 week).
5. Any policy goes through the existing twin/latency replay and hardware checks unchanged.

Where it would be clearly worth it: training from **real-robot data** (DayDreamer regime,
S10) or fitting/finetuning with very few envs - i.e. when samples, not GPU minutes, are
the scarce resource.
