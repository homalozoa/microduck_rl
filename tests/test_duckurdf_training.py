"""Fresh morphology must not silently inherit an old expert or late curriculum."""
import os
import subprocess
import sys


def test_from_scratch_mode_has_no_experts_and_restores_bootstrap_curriculum():
    check = '''
import mjlab.tasks
from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs
from mjlab.tasks.registry import list_tasks, load_rl_cfg, load_env_cfg
assert vs.FROM_SCRATCH and not vs.WARM_START and not vs.ENABLE_EXPERT_BC
assert vs.FELL_OVER_DISABLE_ITER == 500
assert vs.RECOVERY_ECON_KICKIN_ITER == 1200
for task in list_tasks():
    if "VelStand" not in task:
        continue
    assert load_rl_cfg(task).algorithm.bc_cfg is None
    cfg = load_env_cfg(task)
    assert len(cfg.curriculum["action_rate_weight"].params["weight_stages"]) > 1
'''
    subprocess.run([sys.executable, "-c", check], check=True,
                   env={**os.environ, "MICRODUCK_FROM_SCRATCH": "1"})
