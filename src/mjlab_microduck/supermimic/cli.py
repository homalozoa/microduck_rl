"""`train-supermimic` console script: the SuperMimic-style trainer (PPO path untouched).

    uv run train-supermimic --num-envs 64 --max-env-steps 120 --run-name smoke   # smoke test
    uv run train-supermimic --num-envs 4096 --seed 0 --run-name sm-s0            # benchmark arm
"""

from __future__ import annotations

import sys


def main() -> int:
    import tyro

    from mjlab_microduck.supermimic.trainer import SMCfg, Trainer

    cfg = tyro.cli(SMCfg)
    if cfg.task != "Mjlab-Velocity-Flat-MicroDuck":
        raise SystemExit("train-supermimic: only Mjlab-Velocity-Flat-MicroDuck is implemented (analytic rewards are task-specific)")
    Trainer(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
