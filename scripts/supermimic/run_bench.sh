#!/usr/bin/env bash
# Pre-registered benchmark (docs/supermimic/prereg.md): sequential, one GPU, eval after each run.
set -u
cd "$(dirname "$0")/../.."
export WANDB_MODE=disabled
UV=${UV:-uv}
OUT=logs/bench
mkdir -p "$OUT"
for S in ${SEEDS:-0 1 2}; do
  # ---- PPO
  if [ ! -f "$OUT/ppo-s$S.done" ]; then
    echo "[bench] $(date -Is) PPO seed $S start" | tee -a "$OUT/bench.log"
    timeout 10920 $UV run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
      --agent.save-interval 150 --agent.seed "$S" --agent.max-iterations 100000 \
      --agent.logger tensorboard --agent.run-name "bench-ppo-s$S" > "$OUT/ppo-s$S.out" 2>&1
    d=$(ls -dt logs/rsl_rl/velocity/*bench-ppo-s$S | head -1)
    ln -sfn "$(realpath "$d")" "$OUT/ppo-s$S"
    $UV run python scripts/supermimic/eval_velocity.py --run "$OUT/ppo-s$S" --kind ppo > "$OUT/eval-ppo-s$S.out" 2>&1
    $UV run python scripts/supermimic/eval_velocity.py --run "$OUT/ppo-s$S" --kind ppo --initial-stage >> "$OUT/eval-ppo-s$S.out" 2>&1
    touch "$OUT/ppo-s$S.done"
  fi
  # ---- SuperMimic
  if [ ! -f "$OUT/sm-s$S.done" ]; then
    echo "[bench] $(date -Is) SM seed $S start" | tee -a "$OUT/bench.log"
    $UV run train-supermimic --num-envs 4096 --seed "$S" --max-hours 3.0 \
      --log-dir "$OUT" --run-name "sm-s$S" > "$OUT/sm-s$S.out" 2>&1
    $UV run python scripts/supermimic/eval_velocity.py --run "$OUT/sm-s$S" --kind sm > "$OUT/eval-sm-s$S.out" 2>&1
    $UV run python scripts/supermimic/eval_velocity.py --run "$OUT/sm-s$S" --kind sm --initial-stage >> "$OUT/eval-sm-s$S.out" 2>&1
    touch "$OUT/sm-s$S.done"
  fi
done
echo "[bench] $(date -Is) all done" | tee -a "$OUT/bench.log"
