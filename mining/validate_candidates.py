"""Enforce the hard provenance gate on mining/candidates.jsonl.

This is the one piece of the pipeline that isn't just prose instructions
to an agent - a candidate that fails these checks is structurally invalid
regardless of what an agent claims about it. Run it after any Prompt A/B
pass:

    uv run python mining/validate_candidates.py
"""

import json
import sys
from pathlib import Path

_REQUIRED_KEYS = {
    "id",
    "rule_type",
    "sides",
    "symptom",
    "trigger_field",
    "host_dependent",
    "source_url",
    "source_type",
    "source_authority",
    "corroborating_urls",
    "confidence",
    "expressible",
    "status",
    "notes",
}
_VALID_STATUSES = {"candidate", "verified", "rejected", "needs_remining"}
_VALID_CONFIDENCE = {"high", "medium", "low"}
_VALID_SOURCE_AUTHORITY = {"authoritative", "corroborated", "anecdotal", None}
_VALID_RULE_TYPES = {"version_incompat", "env_hazard"}

# needs_remining is the one status allowed to lack a source - it exists
# specifically to carry an unsourced claim until it's re-mined or dropped,
# never to let a real candidate skip the gate.
_STATUSES_REQUIRING_SOURCE = {"candidate", "verified", "rejected"}


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"{path}: file not found"]

    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        where = f"{path}:{lineno}"

        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{where}: invalid JSON ({exc})")
            continue

        errors.extend(f"{where} [{candidate.get('id', '?')}]: {msg}" for msg in _validate_candidate(candidate))

    return errors


def _validate_candidate(candidate: dict) -> list[str]:
    errors = []

    missing = _REQUIRED_KEYS - candidate.keys()
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")
        return errors  # further checks would just KeyError

    status = candidate["status"]
    if status not in _VALID_STATUSES:
        errors.append(f"status {status!r} not in {sorted(_VALID_STATUSES)}")

    if candidate["rule_type"] not in _VALID_RULE_TYPES:
        errors.append(f"rule_type {candidate['rule_type']!r} not in {sorted(_VALID_RULE_TYPES)}")

    if candidate["confidence"] not in _VALID_CONFIDENCE:
        errors.append(f"confidence {candidate['confidence']!r} not in {sorted(_VALID_CONFIDENCE)}")

    if candidate["source_authority"] not in _VALID_SOURCE_AUTHORITY:
        valid = sorted(_VALID_SOURCE_AUTHORITY, key=str)
        errors.append(f"source_authority {candidate['source_authority']!r} not in {valid}")

    # The hard provenance gate: no source_url or no symptom is disqualifying
    # for anything claiming to be an actual candidate.
    if status in _STATUSES_REQUIRING_SOURCE:
        if not candidate.get("source_url"):
            errors.append(f"status {status!r} requires a non-empty source_url")
        if not candidate.get("symptom"):
            errors.append(f"status {status!r} requires a non-empty symptom")

    if not candidate.get("notes") and status in ("rejected", "needs_remining"):
        errors.append(f"status {status!r} requires notes explaining why")

    symptom = candidate.get("symptom") or ""
    if len(symptom.split()) > 15:
        errors.append(f"symptom is {len(symptom.split())} words, over the 15-word limit: {symptom!r}")

    if not candidate.get("sides"):
        errors.append("sides must be non-empty")

    if status == "verified" and not candidate.get("fixed_in") and candidate["confidence"] == "high":
        errors.append("verified + high confidence but no fixed_in - high confidence should usually be fix-linked")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    targets = [repo_root / "mining" / "candidates.jsonl"]
    verified_path = repo_root / "mining" / "rules_verified.jsonl"
    if verified_path.is_file():
        targets.append(verified_path)

    all_errors = []
    for path in targets:
        all_errors.extend(validate_file(path))

    if all_errors:
        print(f"{len(all_errors)} problem(s) found:\n")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("All candidates pass the provenance gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
