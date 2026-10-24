"""Episode-level eval of Velocity-Flat checkpoints (PPO or SuperMimic) — docs/supermimic/prereg.md.

    uv run python scripts/supermimic/eval_velocity.py --run <run dir> --kind ppo|sm [--every N]

For every checkpoint of a run: 1024 envs, one full 20 s episode each on the TRAINING env
cfg (DR, pushes, obs noise and delays on), all curricula forced to their final stage,
deterministic policy (PPO: mean action), fixed seed. Writes <run>/eval.jsonl with
fall_rate, lin_err, yaw_err, ret and the checkpoint's wall clock / env steps.

Wall clock of a PPO checkpoint = mtime(model_N.pt) - mtime(model_0.pt) (model_0 is
written after the first iteration, so this omits one iteration, ~1 s); SM checkpoints
carry their own wall clock in the sidecar JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn

TASK = "Mjlab-Velocity-Flat-MicroDuck"
NUM_ENVS = 1024
SEED = 1234


def ppo_policy(path, device):
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    mlp = nn.Sequential(nn.Linear(61, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 14)).to(device)
    mlp.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")})
    mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
    return lambda o: mlp((o - mean) / (std + 1e-2))  # rsl_rl EmpiricalNormalization (eps 1e-2)


def sm_policy(path, device):
    from mjlab_microduck.supermimic.trainer import Policy, SMCfg

    pol = Policy(SMCfg().hidden).to(device)
    pol.load_state_dict(torch.load(path, map_location=device, weights_only=False)["policy"])
    return pol


def list_checkpoints(run: Path, kind: str):
    out = []
    if kind == "ppo":
        files = sorted(run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
        t0 = os.path.getmtime(run / "model_0.pt")
        for f in files:
            it = int(re.findall(r"\d+", f.name)[0])
            out.append((f, {"wall_s": os.path.getmtime(f) - t0, "iteration": it, "env_steps": (it + 1) * 24 * 4096}))
    else:
        for f in sorted(run.glob("ckpt_*.pt")):
            out.append((f, json.loads(f.with_suffix(".json").read_text())))
    return out


@torch.no_grad()
def evaluate(env, policy, device, counter: int = 10**8):
    env.common_step_counter = counter  # 10**8: every curriculum at its final stage (applied at reset)
    torch.manual_seed(SEED)
    obs, _ = env.reset()
    obs = obs["actor"]
    robot = env.scene["robot"]
    n = env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    ret = torch.zeros(n, device=device)
    lin_sum = torch.zeros(n, device=device)
    yaw_sum = torch.zeros(n, device=device)
    cnt = torch.zeros(n, device=device)
    steps = int(round(env.cfg.episode_length_s / env.step_dt))
    for _ in range(steps):
        cmd = env.command_manager.get_command("twist").clone()
        o, r, term, tout, _ = env.step(policy(obs))
        obs = o["actor"]
        ret += r * alive
        fell |= term & alive
        alive &= ~(term | tout)
        v = robot.data.root_link_lin_vel_b
        w = robot.data.root_link_ang_vel_b
        walking = (torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()) > 0.01
        m = (alive & walking).float()
        lin_sum += torch.norm(cmd[:, :2] - v[:, :2], dim=1) * m
        yaw_sum += (cmd[:, 2] - w[:, 2]).abs() * m
        cnt += m
    ok = cnt > 0
    return {
        "fall_rate": float(fell.float().mean()),
        "lin_err": float((lin_sum[ok] / cnt[ok]).mean()),
        "yaw_err": float((yaw_sum[ok] / cnt[ok]).mean()),
        "ret": float(ret.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", choices=("ppo", "sm"), required=True)
    ap.add_argument("--every", type=int, default=1, help="evaluate every N-th checkpoint (the last one always)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--initial-stage", action="store_true", help="diagnostic: last checkpoint only, curricula at stage 0 -> eval_initial.json")
    args = ap.parse_args()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401

    run = Path(args.run)
    out = run / "eval.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["ckpt"] for line in out.read_text().splitlines() if line.strip()}
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = SEED
    env = ManagerBasedRlEnv(cfg, device=args.device)
    cks = list_checkpoints(run, args.kind)
    if args.initial_stage:
        f, meta = cks[-1]
        pol = ppo_policy(f, args.device) if args.kind == "ppo" else sm_policy(f, args.device)
        rec = {"ckpt": f.name, **meta, **evaluate(env, pol, args.device, counter=0)}
        print(json.dumps(rec), flush=True)
        (run / "eval_initial.json").write_text(json.dumps(rec))
        env.close()
        return
    for i, (f, meta) in enumerate(cks):
        if f.name in done or (i % args.every and i != len(cks) - 1):
            continue
        pol = ppo_policy(f, args.device) if args.kind == "ppo" else sm_policy(f, args.device)
        res = evaluate(env, pol, args.device)
        rec = {"ckpt": f.name, **meta, **res}
        print(json.dumps(rec), flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    env.close()


if __name__ == "__main__":
    main()
