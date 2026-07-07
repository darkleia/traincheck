"""Tests for the Field-based JobSpec and the rule engine's handling of it."""

import dataclasses

import pytest

from traincheck.core import RuleEngine
from traincheck.ir import Field
from traincheck.rules import BUILTIN_RULES
from traincheck.validator import JobSpec, Validator, parse_config

NATIVE_CONFIG = {
    "cluster": {
        "nodes": 64,
        "gpus_per_node": 8,
        "gpu_type": "A100-SXM4-80GB",
        "interconnect": "InfiniBand",
    },
    "nccl": {"version": "2.19.0", "algo": "Ring"},
    "parallelism": {"tensor_parallel": 2, "pipeline_parallel": 4, "data_parallel": 64},
    "environment": {"NCCL_IB_DISABLE": 1},
    "model": {"size_billion_params": 70},
    "data": {"dataloader_workers": 2},
    "checkpoint": {"frequency_steps": 5000},
}


def test_fully_resolved_jobspec_fires_rules_unchanged():
    """A config with every field known should be evaluated exactly like
    before the Field wrapping was introduced: no rule gets skipped.
    """
    result = Validator().validate(NATIVE_CONFIG)

    fired = {v.rule.id for v in result.violations}
    assert fired == {"NCCL-RING-001", "NCCL-IB-001", "DATALOADER-001", "CHECKPOINT-001"}
    assert result.needs_verification == []


def test_unknown_field_routes_to_needs_verification_not_a_violation():
    """A rule that reads an unresolved Field must be neither silently passed
    nor turned into a violation - it belongs in needs_verification.
    """
    spec = JobSpec(
        nodes=Field(64, status="resolved", source="test"),
        gpus_per_node=Field(8, status="resolved", source="test"),
        gpu_type=Field("A100", status="resolved", source="test"),
        interconnect=Field("InfiniBand", status="resolved", source="test"),
        nccl_algo=Field("Ring", status="resolved", source="test"),
        nccl_version=Field(None, status="unknown", reason="NCCL_VERSION not readable from container image"),
        nccl_ib_disable=Field(0, status="resolved", source="test"),
        tensor_parallel=Field(2, status="resolved", source="test"),
        pipeline_parallel=Field(4, status="resolved", source="test"),
        data_parallel=Field(64, status="resolved", source="test"),
        dataloader_workers=Field(8, status="resolved", source="test"),
    )

    engine = RuleEngine()
    for rule in BUILTIN_RULES:
        engine.register(rule)
    result = engine.check(vars(spec))

    assert "NCCL-RING-001" not in {v.rule.id for v in result.violations}
    needs_verification = {nv.rule.id: nv for nv in result.needs_verification}
    assert "NCCL-RING-001" in needs_verification
    assert needs_verification["NCCL-RING-001"].field_name == "nccl_version"
    assert needs_verification["NCCL-RING-001"].reason == "NCCL_VERSION not readable from container image"


def test_parse_config_round_trips_all_fields_as_resolved():
    spec = parse_config(NATIVE_CONFIG)

    for f in dataclasses.fields(spec):
        if f.name == "meta":
            # bookkeeping about the spec, not itself a config-derived Field
            continue
        value = getattr(spec, f.name)
        assert isinstance(value, Field)
        assert value.status == "resolved"
        assert value.source == "native"


def test_field_rejects_unknown_status_without_reason():
    with pytest.raises(ValueError):
        Field(None, status="unknown")
