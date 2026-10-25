#!/usr/bin/env python3
"""Smoke-test every registered MicroDuck task, then train each full recipe.

Run inside the project venv. Uses one process per selected GPU, distinct task
directories, local TensorBoard logs, and fresh policies. Completed smoke gates
can be reused with the recorded source/model fingerprint. An explicit plan
amendment may retain unchanged task gates after an independent configuration
comparison and revalidation of every affected task.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint():
    h = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += sorted((ROOT / "src/mjlab_microduck/robot/duckurdf").rglob("*"))
    paths += [ROOT / "uv.lock", ROOT / "pyproject.toml"]
    for p in paths:
        if p.is_file():
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def write_json(path, value):
    with LOCK:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 7])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["MICRODUCK_FROM_SCRATCH"] = "1"
    import mjlab.tasks  # noqa: F401 - plugin registry
    from mjlab.tasks.registry import list_tasks, load_rl_cfg

    tasks = [task for task in list_tasks() if "MicroDuck" in task]
    # Validate the simplest families first; all base/backlash/rough/roller tasks
    # still appear exactly once in the resulting plan.
    first = ["Mjlab-Velocity-Flat-MicroDuck", "Mjlab-StandUp-Flat-MicroDuck",
             "Mjlab-VelStand-Flat-MicroDuck", "Mjlab-Velocity-Flat-MicroDuck-Rollers"]
    tasks.sort(key=lambda t: (first.index(t) if t in first else len(first), t))
    campaign = ROOT / "logs/duckurdf_campaigns" / args.campaign
    campaign.mkdir(parents=True, exist_ok=True)
    signature = fingerprint()
    plan = dict(created=now(), source_fingerprint=signature, gpus=args.gpus,
                training_mode="from_scratch_without_old_morphology_experts",
                num_envs=args.num_envs, tasks=[dict(task=t, recipe=load_rl_cfg(t).experiment_name,
                                                  iterations=load_rl_cfg(t).max_iterations) for t in tasks])
    plan_file = campaign / "plan.json"
    if plan_file.exists():
        previous = json.loads(plan_file.read_text())
        if any(previous[k] != plan[k] for k in ("source_fingerprint", "gpus", "num_envs", "tasks")):
            raise RuntimeError("Campaign source/settings changed; choose a new campaign name")
    else:
        write_json(plan_file, plan)
    status = dict(phase="smoke", started=now(), pid=os.getpid(), total_tasks=len(tasks), tasks={})
    status_file = campaign / "status.json"
    if status_file.exists():
        status["tasks"] = json.loads(status_file.read_text()).get("tasks", {})

    def update(task, phase, state, **details):
        with LOCK:
            status["tasks"].setdefault(task, {})[phase] = dict(state=state, updated=now(), **details)
        write_json(status_file, status)
        print(f"[{now()}] {phase} {state}: {task} {details}", flush=True)

    def run(task, gpu, phase):
        if status["tasks"].get(task, {}).get(phase, {}).get("state") == "passed":
            return True
        if fingerprint() != signature:
            raise RuntimeError("Source/model files changed during campaign")
        slug = task.removeprefix("Mjlab-")
        experiment = f"duckurdf/{args.campaign}/{phase}/{slug}"
        stdout = campaign / f"{phase}_{slug}.log"
        command = [sys.executable, "-u", "-m", "mjlab_microduck.train_cli", task,
                   "--env.scene.num-envs", str(64 if phase == "smoke" else args.num_envs),
                   "--agent.logger", "tensorboard", "--agent.upload-model", "False",
                   "--agent.resume", "False", "--agent.experiment-name", experiment,
                   "--agent.run-name", "duckurdf", "--enable-nan-guard", "True"]
        if phase == "smoke":
            command += ["--agent.max-iterations", "5", "--agent.save-interval", "5"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
               "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MUJOCO_GL": "egl",
               "WANDB_MODE": "disabled"}
        env.pop("MICRODUCK_WARM_START", None)
        update(task, phase, "running", gpu=gpu, log=str(stdout), command=command)
        with stdout.open("a") as out:
            result = subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    timeout=1800 if phase == "smoke" else None)
            if result.returncode:
                update(task, phase, "failed", gpu=gpu, returncode=result.returncode, log=str(stdout))
                return False
            checkpoints = sorted((ROOT / "logs/rsl_rl" / experiment).glob("*/model_*.pt"),
                                 key=lambda p: p.stat().st_mtime)
            if not checkpoints:
                update(task, phase, "failed", gpu=gpu, error="No checkpoint", log=str(stdout))
                return False
            checkpoint = checkpoints[-1]
            onnx = checkpoint.parent / "policy.onnx"
            export = [sys.executable, "-u", "scripts/export.py", task,
                      "--checkpoint-file", str(checkpoint), "--onnx-file", str(onnx),
                      "--num-envs", "1", "--device", "cuda:0"]
            result = subprocess.run(export, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            update(task, phase, "failed", gpu=gpu, error="ONNX export failed", log=str(stdout))
            return False
        import numpy as np
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        assert len(inputs) == len(outputs) == 1
        assert inputs[0].shape == [1, 61], inputs[0].shape
        assert outputs[0].shape == [1, 14], outputs[0].shape
        output = session.run(None, {inputs[0].name: np.zeros((1, 61), dtype=np.float32)})[0]
        assert np.isfinite(output).all()
        update(task, phase, "passed", gpu=gpu, checkpoint=str(checkpoint), onnx=str(onnx), log=str(stdout))
        return True

    def phase(name):
        status["phase"] = name
        write_json(status_file, status)
        pending = queue.Queue()
        for task in tasks:
            pending.put(task)
        failures = []

        def worker(gpu):
            while True:
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    if not run(task, gpu, name):
                        failures.append(task)
                except Exception as exc:
                    update(task, name, "failed", gpu=gpu, error=repr(exc))
                    failures.append(task)
                finally:
                    pending.task_done()
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            list(pool.map(worker, args.gpus))
        return failures

    failed = phase("smoke")
    if failed:
        status.update(phase="smoke_failed", finished=now(), failures=failed)
        write_json(status_file, status)
        return 1
    if args.smoke_only:
        status.update(phase="smoke_complete", finished=now())
        write_json(status_file, status)
        return 0
    failed = phase("train")
    status.update(phase="train_failed" if failed else "complete", finished=now(), failures=failed)
    write_json(status_file, status)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
