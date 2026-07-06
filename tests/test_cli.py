"""CLI smoke tests for `traincheck check`."""

import json
from pathlib import Path

from typer.testing import CliRunner

from traincheck.cli import app

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "slurm"

runner = CliRunner()

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
