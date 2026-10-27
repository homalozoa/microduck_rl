# Results: SuperMimic-style trainer vs PPO, Mjlab-Velocity-Flat-MicroDuck

Pre-registration: `prereg.md` (commit 158b03c, before any benchmark run). Code: branch
`supermimic` at a30cdf3 for every run. Runs: linkff, 1× RTX 4090, sequential, 2026-10-24
21:08 to 2026-10-25 19:04 (CST). Raw data on linkff: `~/microduck_rl-supermimic/logs/bench/`
(`*/eval.jsonl`, `*/eval_initial.json`, `*.out`).

## Verdict (pre-registered rules)

**No clear speedup. By the letter of the rules the verdict is NEGATIVE**: the 3-seed SM
final fall rate (0.106) is worse than PPO's (0.052) beyond the +0.02 margin. That margin
is breached entirely by seed 2, which ran on a contaminated machine (see Deviations).
Excluding seed 2, the result is **parity / no clear speedup**: SM reaches the criterion
about 2× *later* in wall clock than PPO, then ends better.

## Usable-gait criterion
- **Absolute criterion (fall ≤ 0.05, lin_err ≤ 0.10, yaw_err ≤ 0.25):** no run of either
  arm met it. lin_err plateaus at about 0.20 m/s for PPO and about 0.18 m/s for SM.
- **Relative criterion, therefore used as pre-registered:** fall ≤ PPO_final + 0.02 =
  0.0718 AND lin_err ≤ 1.10 × PPO_final = 0.2246 (PPO_final = 3-seed mean of the last
  checkpoint: fall 0.0518, lin 0.2042). Two consecutive checkpoints must pass.

| run | T_use (min) | N_use (env steps) | env steps in 3 h |
|---|---|---|---|
| PPO s0 | 53.2 | 310 M | 1047 M |
| PPO s1 | 55.6 | 325 M | 1047 M |
| PPO s2 | 42.7 | 251 M | 988 M |
| SM s0 | 100.1 | 153 M | 278 M |
| SM s1 | 105.1 | 165 M | 267 M |
| SM s2 | not reached by 180 (censored) | - | 146 M (machine contended) |

Median T_use: PPO 53.2 min, SM 105.1 min, so SM is 1.98× slower (speedup criterion was
≤ 0.67×). Env steps to criterion: SM needs about half of PPO's (153–165 M vs 251–325 M).

## Final checkpoint (3 h), final-stage eval
| run | fall | lin_err | yaw_err | ret |
|---|---|---|---|---|
| PPO s0 | 0.053 | 0.205 | 0.284 | 152.3 |
| PPO s1 | 0.040 | 0.204 | 0.270 | 152.8 |
| PPO s2 | 0.062 | 0.204 | 0.294 | 151.6 |
| SM s0 | 0.019 | 0.179 | 0.167 | 163.3 |
| SM s1 | 0.018 | 0.182 | 0.171 | 162.3 |
| SM s2 | 0.282 | 0.201 | 0.292 | 130.4 |

Initial-stage diagnostic (curricula at stage 0, last checkpoint; not part of the verdict):
| run | fall | lin_err | yaw_err |
|---|---|---|---|
| PPO s0 | 0.020 | 0.196 | 0.255 |
| PPO s1 | 0.014 | 0.197 | 0.223 |
| PPO s2 | 0.021 | 0.195 | 0.237 |
| SM s0 | 0.008 | 0.168 | 0.115 |
| SM s1 | 0.008 | 0.172 | 0.122 |
| SM s2 | 0.010 | 0.167 | 0.122 |

## SM time split and world model (end of run)
- s0: sim 2765 s / world-model fitting 4139 s / policy+critic 3891 s; 27 134 updates.
- Model error at the end of the run: nse1 about 0.0008, nse8 about 0.005; contact accuracy 0.99.
- Imagined survival over H = 32 is 0.999.

## Deviations from the pre-registration
1. **SM s2 ran on a loaded machine.** The prereg requires no other job on the GPU host.
   linkff's sysstat shows another workload during SM s2 (CPU 94–96% busy, load average
   47–68 on 32 cores at 15:00 and 16:00, against about 3% idle-time usage otherwise).
   SM s2's throughput was half of s0/s1 for its whole 3 h: 35.6 k env steps vs 67.9 k /
   65.3 k. CPU-busy periods also hit 10:00–12:00 (end of SM s1, its eval, start of
   PPO s2), but PPO s2's iteration time stayed at 1.0 s. SM s2 is therefore not a valid
   measurement of wall clock. Its initial-stage diagnostic matches s0/s1, which suggests
   it simply had less compute. A rerun of SM s2 on an idle machine is the fix. Without
   it, the pre-registered verdict is the NEGATIVE one stated above.
2. PPO s2's last checkpoint ≤ 3 h is iteration 10050 at 178.7 min (s0/s1: 10650 at 179.6
   min).
3. The pre-registered absolute criterion turned out unreachable for both arms. The
   relative criterion was used, as the prereg provides.

## Why SM is slower here (interpretation, not part of the verdict)
- **Throughput.** On one 4090, PPO collects about 94 k env steps/s. SM collects about
  25 k: each update spends about 0.15 s fitting the world model and about 0.14 s on
  BPTT, at 0.4 updates per env step. SM needs about 2× fewer env steps to reach the
  criterion, but runs at 1/4 of the env-step rate, so it is about 2× slower in wall clock.
- **Curricula are keyed on env steps.** The final stage (2000 × 24 steps) comes after
  about 35 min for PPO and after about 2.2 h for SM. SM's eval curve only drops once the
  final stage arrives (falls 1.0 until about 45 min, 0.16 at 90 min).
- **Quality at 3 h.** SM s0/s1 end better than every PPO seed on all four metrics, with
  about 1/4 of the samples. The largest difference is yaw_err (0.17 vs 0.27–0.29). A
  deterministic policy trained on exact analytic rewards plus a learned model is a
  plausible reason; this benchmark does not test it.

## Not done
- SitStand is not proposed, because there was no clear speedup.
- Sim-to-real was not checked, nor was ONNX export of the SM policy.
