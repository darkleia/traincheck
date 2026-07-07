"""Tests for mining/validate_candidates.py - the hard provenance gate the
mining pipeline runs after every Prompt A/B pass (see mining/README.md).

This isn't testing traincheck itself; it's proving the gate is actually a
gate and not just a script that always prints "pass" - every case here is a
candidate the mining pipeline could plausibly produce by mistake, and each
one must be rejected before we trust the gate to guard real promotions.
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "mining" / "validate_candidates.py"
_spec = importlib.util.spec_from_file_location("validate_candidates", _MODULE_PATH)
validate_candidates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_candidates)

_validate_candidate = validate_candidates._validate_candidate
validate_file = validate_candidates.validate_file

_VALID_CANDIDATE = {
    "id": "example-001",
    "rule_type": "version_incompat",
    "sides": [{"component": "nccl", "version_range": ">=2.18.1,<2.18.3"}],
    "trigger_field": "nccl_version",
    "host_dependent": False,
    "source_url": "https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-18-1.html",
    "source_type": "nvidia_release_notes",
    "source_authority": "authoritative",
    "corroborating_urls": [],
    "confidence": "medium",
    "expressible": True,
    "status": "candidate",
    "notes": "",
    "symptom": "AllReduce data corruption on H100",
}


def _candidate(**overrides) -> dict:
    candidate = dict(_VALID_CANDIDATE)
    candidate.update(overrides)
    return candidate


def test_a_genuinely_valid_candidate_passes_with_no_errors():
    """Sanity check the fixture itself, so a failure elsewhere can't be
    blamed on a broken baseline.
    """
    assert _validate_candidate(_candidate()) == []


def test_gate_rejects_a_candidate_with_no_source_url():
    errors = _validate_candidate(_candidate(source_url=None))
    assert any("source_url" in e for e in errors)


def test_gate_rejects_a_verified_claim_with_no_source_url():
    errors = _validate_candidate(_candidate(status="verified", source_url=None))
    assert any("source_url" in e for e in errors)


def test_gate_rejects_a_symptom_over_the_15_word_limit():
    long_symptom = " ".join(["word"] * 16)
    errors = _validate_candidate(_candidate(symptom=long_symptom))
    assert any("15-word limit" in e for e in errors)


def test_gate_rejects_missing_required_keys():
    incomplete = {"id": "broken", "status": "candidate"}
    errors = _validate_candidate(incomplete)
    assert any("missing required keys" in e for e in errors)


def test_gate_rejects_an_invalid_status():
    errors = _validate_candidate(_candidate(status="probably-fine"))
    assert any("status" in e and "not in" in e for e in errors)


def test_gate_rejects_an_invalid_confidence():
    errors = _validate_candidate(_candidate(confidence="extremely-high"))
    assert any("confidence" in e for e in errors)


def test_gate_rejects_rejected_status_with_no_notes():
    errors = _validate_candidate(_candidate(status="rejected", notes=""))
    assert any("requires notes" in e for e in errors)


def test_gate_rejects_empty_sides():
    errors = _validate_candidate(_candidate(sides=[]))
    assert any("sides must be non-empty" in e for e in errors)


def test_gate_rejects_high_confidence_verified_with_no_fixed_in_for_a_bug_report():
    errors = _validate_candidate(
        _candidate(status="verified", confidence="high", fixed_in=None, source_type="github_issue_with_fix")
    )
    assert any("fixed_in" in e for e in errors)


def test_gate_exempts_vendor_matrix_from_the_fixed_in_check():
    """A published vendor compatibility requirement (e.g. minimum driver
    version) isn't a bug with a fix - it's exempt from the fix-linkage
    check a bug-report candidate would otherwise need for high confidence.
    """
    errors = _validate_candidate(
        _candidate(status="verified", confidence="high", fixed_in=None, source_type="vendor_matrix")
    )
    assert errors == []


def test_needs_remining_is_the_only_status_allowed_to_skip_source_url():
    errors = _validate_candidate(_candidate(status="needs_remining", source_url=None, notes="still looking"))
    assert errors == []


def test_validate_file_reports_invalid_json_with_line_number():
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "ok"\n')  # truncated/invalid JSON
        path = Path(f.name)

    try:
        errors = validate_file(path)
        assert any("invalid JSON" in e for e in errors)
        assert any(f"{path}:1" in e for e in errors)
    finally:
        path.unlink()


def test_the_real_candidates_file_currently_passes_the_gate():
    """Not a fixed-data test: confirms the actual mining/candidates.jsonl
    and mining/rules_verified.jsonl in this repo pass right now, so this
    test fails loudly the moment a bad entry is added instead of silently
    bit-rotting.
    """
    repo_root = Path(__file__).resolve().parent.parent
    errors = validate_file(repo_root / "mining" / "candidates.jsonl")
    errors += validate_file(repo_root / "mining" / "rules_verified.jsonl")
    assert errors == []
