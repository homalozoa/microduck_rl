"""SuperMimic-style trainer (learned world model + BPTT actor) for microduck velocity tasks.

Alternative to the PPO runner, reached only through the ``train-supermimic`` entry point;
nothing here is imported by the PPO path. The world-model / imagination / critic structure
follows /Users/homalozoax/bfm/code/supermimic_duck/smduck, itself a port of
CuboiLeo/SuperMimic-Reproduced @d9be385 (MIT, see LICENSE.SuperMimic-Reproduced).
``rotations.py`` and ``fk.py`` are copied from there; everything else is new for the
reward-based velocity task. Design and benchmark: docs/supermimic/.
"""
