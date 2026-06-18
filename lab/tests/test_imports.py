"""Smoke: every shipped module imports cleanly.

CI runs pytest only, so a syntax/import break in a module the unit tests don't
touch (studio, the eval baselines, voices) would otherwise slip through. Heavy
optional deps (basic_pitch) are imported lazily inside their functions, so the
module import itself stays dependency-free.
"""

import importlib

import pytest

MODULES = [
    "mambo_lab.ir", "mambo_lab.actions", "mambo_lab.melody", "mambo_lab.router",
    "mambo_lab.fuse", "mambo_lab.oracle", "mambo_lab.planner", "mambo_lab.probe",
    "mambo_lab.speech", "mambo_lab.percussion", "mambo_lab.cli", "mambo_lab.studio",
    "mambo_lab.session", "mambo_lab.semantic_verify", "mambo_lab.settings", "mambo_lab.voiceprint", "mambo_lab.daw.reaper",
    "mambo_lab.eval.gate", "mambo_lab.eval.metrics", "mambo_lab.eval.voices",
    "mambo_lab.eval.elive_baselines", "mambo_lab.eval.basicpitch_baseline",
    "mambo_lab.eval.ablation", "mambo_lab.eval.b6_omni", "mambo_lab.eval.b6_qwen", "mambo_lab.eval.multiseed",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)
