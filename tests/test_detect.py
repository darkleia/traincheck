"""Tests for the full stack-detection signature table."""

from pathlib import Path

import pytest

from traincheck.detect import Stack, detect_stack

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPO_ROOT / "examples"

ENTRYPOINT_BY_STACK = {
    Stack.SLURM: EXAMPLES_ROOT / "slurm" / "train.sbatch",
    Stack.LSF: EXAMPLES_ROOT / "lsf" / "train.lsf",
    Stack.PBS: EXAMPLES_ROOT / "pbs" / "train.pbs",
    Stack.SGE: EXAMPLES_ROOT / "sge" / "train.sge",
    Stack.K8S_CRD: EXAMPLES_ROOT / "k8s_crd" / "pytorchjob.yaml",
    Stack.RAY: EXAMPLES_ROOT / "ray" / "cluster.yaml",
    Stack.SKYPILOT: EXAMPLES_ROOT / "skypilot" / "task.yaml",
    Stack.BARE: EXAMPLES_ROOT / "bare" / "run.sh",
    Stack.TORCHX: EXAMPLES_ROOT / "torchx" / "component.py",
    Stack.SUBMITIT: EXAMPLES_ROOT / "submitit" / "job.py",
    Stack.NATIVE: EXAMPLES_ROOT / "native" / "job.traincheck.yaml",
}


@pytest.mark.parametrize("expected_stack,path", ENTRYPOINT_BY_STACK.items())
def test_detect_stack_matches_every_example_entrypoint(expected_stack, path):
    assert detect_stack(path) == expected_stack


def test_detect_stack_falls_back_to_unknown_for_garbage(tmp_path):
    garbage = tmp_path / "notes.txt"
    garbage.write_text("just some notes, not a config of any kind\n")

    assert detect_stack(garbage) == Stack.UNKNOWN


def test_detect_stack_raises_and_lists_candidates_for_a_directory():
    with pytest.raises(IsADirectoryError) as excinfo:
        detect_stack(EXAMPLES_ROOT / "slurm")

    message = str(excinfo.value)
    assert "train.sbatch" in message
    assert "ds_config.json" in message
