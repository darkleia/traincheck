"""CLI smoke tests for `traincheck check`."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from traincheck.cli import app

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPO_ROOT / "examples"
EXAMPLES_DIR = EXAMPLES_ROOT / "slurm"

runner = CliRunner()

# One entrypoint per supported stack - same files resolve() recognizes
# (see test_resolve_stacks.py for why torchx's is component.py, not run.sh).
ALL_STACK_ENTRYPOINTS = [
    EXAMPLES_ROOT / "slurm" / "train.sbatch",
    EXAMPLES_ROOT / "k8s_crd" / "pytorchjob.yaml",
    EXAMPLES_ROOT / "skypilot" / "task.yaml",
    EXAMPLES_ROOT / "ray" / "cluster.yaml",
    EXAMPLES_ROOT / "bare" / "run.sh",
    EXAMPLES_ROOT / "torchx" / "component.py",
    EXAMPLES_ROOT / "submitit" / "job.py",
    EXAMPLES_ROOT / "native" / "job.traincheck.yaml",
]

# Triggers NCCL-RING-001 (an ERROR-severity rule): Ring algo, >32 nodes,
# an A100 GPU type, and an NCCL version older than 2.21 - all fields the
# Slurm+shell adapter can resolve directly from the header/body alone.
BROKEN_SBATCH = """\
#!/bin/bash
#SBATCH --job-name=broken-train
#SBATCH --nodes=64
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=A100-SXM4-80GB

module load cuda/12.2
module load nccl/2.19

export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=0

srun torchrun --nnodes=64 --nproc-per-node=8 train.py
"""


def test_cli_check_prints_three_sections_for_clean_sbatch_fixture():
    result = runner.invoke(app, ["check", str(EXAMPLES_DIR / "train.sbatch")])

    assert "Violations" in result.stdout
    assert "Needs verification" in result.stdout
    assert "Summary" in result.stdout
    # a known host-fact check command should show up in the human output
    assert "lsmod | grep peermem" in result.stdout
    assert result.exit_code == 0


def test_cli_check_exits_1_when_a_real_error_violation_fires(tmp_path):
    broken = tmp_path / "broken.sbatch"
    broken.write_text(BROKEN_SBATCH)

    result = runner.invoke(app, ["check", str(broken)])

    assert result.exit_code == 1
    assert "NCCL-RING-001" in result.stdout


def test_cli_check_json_includes_needs_verification_array():
    result = runner.invoke(app, ["check", str(EXAMPLES_DIR / "train.sbatch"), "--json"])

    payload = json.loads(result.stdout)
    assert "needs_verification" in payload
    assert any(item["field"] == "peermem_loaded" for item in payload["needs_verification"])
    assert all("check_command" in item for item in payload["needs_verification"])


@pytest.mark.parametrize("entrypoint", ALL_STACK_ENTRYPOINTS, ids=lambda p: p.parent.name)
def test_cli_check_prints_three_sections_for_every_stack(entrypoint):
    result = runner.invoke(app, ["check", str(entrypoint)])

    assert result.exception is None, f"{entrypoint}: unhandled exception: {result.output}"
    assert "Violations" in result.stdout
    assert "Needs verification" in result.stdout
    assert "Summary" in result.stdout
    assert result.exit_code in (0, 1)


@pytest.mark.parametrize("entrypoint", ALL_STACK_ENTRYPOINTS, ids=lambda p: p.parent.name)
def test_cli_check_json_is_valid_for_every_stack(entrypoint):
    result = runner.invoke(app, ["check", str(entrypoint), "--json"])

    payload = json.loads(result.stdout)
    assert "passed" in payload
    assert "violations" in payload
    assert "needs_verification" in payload


def test_cli_check_reports_a_clean_error_for_an_unsupported_stack(tmp_path):
    """A file resolve() can't route to any adapter must produce a plain
    error message, not an uncaught traceback.
    """
    empty = tmp_path / "empty.yaml"
    empty.write_text("")

    result = runner.invoke(app, ["check", str(empty)])

    assert result.exit_code == 2
    assert "doesn't support this stack yet" in result.output
    assert "Traceback" not in result.output


def test_cli_check_reports_a_clean_error_for_a_scheduler_without_an_adapter():
    result = runner.invoke(
        app, ["check", str(EXAMPLES_ROOT / "lsf" / "train.lsf")]
    )

    assert result.exit_code == 2
    assert "doesn't support this stack yet" in result.output
    assert "Traceback" not in result.output
