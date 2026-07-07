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
    assert fired == {"NCCL-IB-001", "DATALOADER-001", "CHECKPOINT-001"}
    assert result.needs_verification == []


def test_unknown_field_routes_to_needs_verification_not_a_violation():
    """A rule that reads an unresolved Field must be neither silently passed
    nor turned into a violation - it belongs in needs_verification.
    """
    spec = JobSpec(
        gpu_type=Field("H100", status="resolved", source="test"),
        nccl_net_gdr_level=Field(None, status="unknown", reason="NCCL_NET_GDR_LEVEL not readable from container image"),
    )

    engine = RuleEngine()
    for rule in BUILTIN_RULES:
        engine.register(rule)
    result = engine.check(vars(spec))

    assert "NCCL-GDR-001" not in {v.rule.id for v in result.violations}
    needs_verification = {nv.rule.id: nv for nv in result.needs_verification}
    assert "NCCL-GDR-001" in needs_verification
    assert needs_verification["NCCL-GDR-001"].field_name == "nccl_net_gdr_level"
    assert needs_verification["NCCL-GDR-001"].reason == "NCCL_NET_GDR_LEVEL not readable from container image"


def test_a_condition_using_str_can_actually_fire():
    """Regression test: Rule.evaluate used to run conditions with no
    builtins at all, so any condition calling str(...) (like NCCL-GDR-001's
    str(gpu_type).startswith('H100')) raised NameError internally on every
    evaluation - silently swallowed into "condition is False" by evaluate's
    broad except, so the rule looked registered and correct but could never
    actually fire.
    """
    spec = JobSpec(
        gpu_type=Field("H100-SXM5-80GB", status="resolved", source="test"),
        nccl_net_gdr_level=Field(3, status="resolved", source="test"),
    )

    engine = RuleEngine()
    for rule in BUILTIN_RULES:
        engine.register(rule)
    result = engine.check(vars(spec))

    assert "NCCL-GDR-001" in {v.rule.id for v in result.violations}


def test_parse_config_round_trips_all_fields_as_resolved():
    spec = parse_config(NATIVE_CONFIG)

    for f in dataclasses.fields(spec):
        if f.name in ("meta", "comm_env"):
            # bookkeeping / a dict-of-Fields bucket, not itself a single
            # config-derived Field - see test_parse_config_comm_env below
            continue
        value = getattr(spec, f.name)
        assert isinstance(value, Field)
        assert value.status == "resolved"
        assert value.source == "native"


def test_parse_config_comm_env_reads_from_the_environment_block():
    spec = parse_config(NATIVE_CONFIG)

    assert spec.comm_env["NCCL_IB_DISABLE"].status == "resolved"
    assert spec.comm_env["NCCL_IB_DISABLE"].value == 1
    assert spec.comm_env["NCCL_IB_DISABLE"].source == "native"
    # not present in NATIVE_CONFIG's "environment" block
    assert spec.comm_env["NCCL_SOCKET_IFNAME"].status == "absent"


def test_field_rejects_unknown_status_without_reason():
    with pytest.raises(ValueError):
        Field(None, status="unknown")
