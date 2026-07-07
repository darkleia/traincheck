"""CLI smoke tests for `traincheck check`."""

import json
from pathlib import Path
from unittest.mock import patch

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
    EXAMPLES_ROOT / "pbs" / "train.pbs",
    EXAMPLES_ROOT / "lsf" / "train.lsf",
    EXAMPLES_ROOT / "sge" / "train.sge",
    EXAMPLES_ROOT / "k8s_crd" / "pytorchjob.yaml",
    EXAMPLES_ROOT / "trainjob" / "trainjob.yaml",
    EXAMPLES_ROOT / "skypilot" / "task.yaml",
    EXAMPLES_ROOT / "accelerate" / "default_config.yaml",
    EXAMPLES_ROOT / "ray" / "cluster.yaml",
    EXAMPLES_ROOT / "bare" / "run.sh",
    EXAMPLES_ROOT / "torchx" / "component.py",
    EXAMPLES_ROOT / "submitit" / "job.py",
    EXAMPLES_ROOT / "native" / "job.traincheck.yaml",
]

# Triggers PARALLEL-002 (an ERROR-severity rule): the DeepSpeed config's
# tensor_parallel_size * pipeline_parallel_size (3) doesn't evenly divide
# the header's own world_size (5 nodes * 2 GPUs/node = 10) - all fields
# the Slurm+shell+DeepSpeed adapter can resolve directly.
BROKEN_SBATCH = """\
#!/bin/bash
#SBATCH --job-name=broken-train
#SBATCH --nodes=5
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

srun torchrun --nnodes=5 --nproc-per-node=2 train.py --deepspeed ds_config.json
"""

BROKEN_DS_CONFIG = """\
{
  "tensor_parallel_size": 3,
  "pipeline_parallel_size": 1
}
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
    (tmp_path / "ds_config.json").write_text(BROKEN_DS_CONFIG)

    result = runner.invoke(app, ["check", str(broken)])

    assert result.exit_code == 1
    assert "PARALLEL-002" in result.stdout


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


def _fake_completed(stdout: str, returncode: int = 0):
    return type("Completed", (), {"stdout": stdout, "returncode": returncode})()


def test_cli_check_without_probe_host_has_no_host_facts_section():
    result = runner.invoke(app, ["check", str(EXAMPLES_DIR / "train.sbatch")])

    assert "Host facts" not in result.stdout


def test_cli_check_probe_host_resolves_and_shows_host_facts():
    def fake_run(cmd, **kwargs):
        outputs = {
            "nvidia-smi": "535.129.03\n",
            "uname": "5.15.0-generic\n",
            "ofed_info": "MLNX_OFED_LINUX-5.8-1.0.1.1\n",
            "lsmod": "nvidia_peermem 16384 0\n",
        }
        return _fake_completed(outputs.get(cmd[0], ""))

    with patch("traincheck.hostprobe.subprocess.run", side_effect=fake_run):
        result = runner.invoke(app, ["check", str(EXAMPLES_DIR / "train.sbatch"), "--probe-host"])

    assert "Host facts (probed on this machine)" in result.stdout
    assert "535.129.03" in result.stdout
    assert "5.15.0-generic" in result.stdout
    # all four resolved, so nothing host-related should remain in "Needs verification"
    assert "verify NVIDIA driver version" not in result.stdout
    assert result.exit_code == 0


def test_cli_check_probe_host_json_includes_host_facts():
    def fake_run(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            return _fake_completed("535.129.03\n")
        return _fake_completed("", returncode=1)

    with patch("traincheck.hostprobe.subprocess.run", side_effect=fake_run):
        result = runner.invoke(app, ["check", str(EXAMPLES_DIR / "train.sbatch"), "--probe-host", "--json"])

    payload = json.loads(result.stdout)
    assert payload["host_facts"]["driver_version"] == {"status": "resolved", "value": "535.129.03"}
    assert payload["host_facts"]["kernel_version"]["status"] == "unknown"


def test_cli_check_probe_host_still_lists_unresolvable_facts():
    with patch("traincheck.hostprobe.subprocess.run", side_effect=FileNotFoundError):
        result = runner.invoke(app, ["check", str(EXAMPLES_DIR / "train.sbatch"), "--probe-host"])

    assert "Host facts (probed on this machine)" in result.stdout
    assert "still unknown" in result.stdout
