"""Tests for the Phase 1 stack detector."""

from pathlib import Path

from traincheck.detect import Stack, detect_stack

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "slurm"


def test_detect_stack_recognizes_sbatch_script():
    assert detect_stack(EXAMPLES_DIR / "train.sbatch") == Stack.SLURM


def test_detect_stack_recognizes_native_schema():
    assert detect_stack(REPO_ROOT / "test_config.yaml") == Stack.NATIVE


def test_detect_stack_falls_back_to_unknown():
    assert detect_stack(EXAMPLES_DIR / "train.py") == Stack.UNKNOWN
