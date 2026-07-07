"""Tests for mined-and-verified version-incompatibility rules (see
mining/README.md and mining/rules_verified.jsonl for how each one here was
sourced - nothing in rules/version_incompat.py may be authored from
memory).
"""

from traincheck.core import RuleEngine
from traincheck.ir import Field
from traincheck.rules.version_incompat import VERSION_INCOMPAT_RULES
from traincheck.validator import JobSpec


def _fired_ids(spec: JobSpec) -> set:
    engine = RuleEngine()
    for rule in VERSION_INCOMPAT_RULES:
        engine.register(rule)
    result = engine.check(vars(spec))
    return {v.rule.id for v in result.violations}


def _spec(nccl_version, gpu_type) -> JobSpec:
    spec = JobSpec()
    spec.nccl_version = Field(nccl_version, status="resolved", source="test")
    spec.gpu_type = Field(gpu_type, status="resolved", source="test")
    return spec


def test_nccl_h100_001_fires_in_the_known_broken_range():
    assert "NCCL-H100-001" in _fired_ids(_spec((2, 18, 1), "H100-SXM5-80GB"))
    assert "NCCL-H100-001" in _fired_ids(_spec((2, 18, 2), "H100"))


def test_nccl_h100_001_does_not_fire_once_fixed():
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 18, 3), "H100"))
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 19, 0), "H100"))


def test_nccl_h100_001_does_not_fire_before_the_broken_range():
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 18, 0), "H100"))


def test_nccl_h100_001_does_not_fire_on_other_gpu_types():
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 18, 1), "A100-SXM4-80GB"))


def test_nccl_h100_001_message_cites_its_source():
    engine = RuleEngine()
    for rule in VERSION_INCOMPAT_RULES:
        engine.register(rule)
    result = engine.check(vars(_spec((2, 18, 1), "H100")))

    violation = next(v for v in result.violations if v.rule.id == "NCCL-H100-001")
    assert "docs.nvidia.com" in violation.rule.message
