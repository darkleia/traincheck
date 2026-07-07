"""Tests for stack dispatch (resolve) and the end-to-end Slurm pipeline."""

from pathlib import Path

import pytest

from traincheck.resolve import UnsupportedStackError, resolve
from traincheck.validator import JobSpec, Validator
from traincheck.verification import collect_needs_verification

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "slurm"
NATIVE_CONFIG_PATH = REPO_ROOT / "examples" / "native" / "job.traincheck.yaml"


def test_resolve_dispatches_native_yaml_to_parse_config():
    spec = resolve(str(NATIVE_CONFIG_PATH))

    assert isinstance(spec, JobSpec)
    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8


def test_resolve_dispatches_sbatch_to_adapt_slurm():
    spec = resolve(str(EXAMPLES_DIR / "train.sbatch"))

    assert isinstance(spec, JobSpec)
    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8


def test_resolve_raises_clear_error_for_unsupported_stack():
    with pytest.raises(UnsupportedStackError):
        resolve(str(EXAMPLES_DIR / "ds_config.json"))


def test_end_to_end_slurm_pipeline_rules_all_evaluate():
    """Neither PARALLEL-001 nor NCCL-GDR-001 should be blocked as
    needs_verification - every field their conditions reference is
    resolved (or genuinely absent, never unknown) by the Slurm+shell+
    DeepSpeed pipeline, so the rule engine reaches a real verdict.
    """
    spec = resolve(str(EXAMPLES_DIR / "train.sbatch"))
    result = Validator().validate_spec(spec)

    blocked_rule_ids = {nv.rule.id for nv in result.needs_verification}
    for rule_id in ("PARALLEL-001", "NCCL-GDR-001"):
        assert rule_id not in blocked_rule_ids


def test_gdr_and_ib_host_dependencies_surface_as_needs_verification_with_commands():
    spec = resolve(str(EXAMPLES_DIR / "train.sbatch"))
    result = Validator().validate_spec(spec)

    items = {item.field_name: item for item in collect_needs_verification(spec, result)}

    assert "peermem_loaded" in items  # GDR (GPUDirect RDMA) host dependency
    assert items["peermem_loaded"].check_command == "lsmod | grep peermem"

    assert "ofed_version" in items  # InfiniBand driver-stack host dependency
    assert items["ofed_version"].check_command == "ofed_info -s"
