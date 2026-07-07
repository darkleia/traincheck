"""Small helpers shared across parsers/adapters."""

import re
from pathlib import Path
from typing import Any, Optional

import yaml


def parse_version(version: Optional[str]) -> Optional[tuple]:
    """Turn a dotted version string like "2.19" into (2, 19)."""
    if version is None:
        return None
    try:
        return tuple(int(x) for x in version.split("."))
    except (ValueError, AttributeError):
        return None


_LOOSE_CONSTRAINT_RE = re.compile(r"[<>~!,]")


def parse_pinned_version(constraint: Optional[str]) -> Optional[tuple]:
    """Turn a dependency constraint into a version tuple, but only when it
    names exactly one version: an "==X.Y.Z" pin (as requirements.txt or
    environment.yml would write it) or a bare "X.Y.Z" (as an already-resolved
    lock file like uv.lock/poetry.lock/Pipfile.lock stores it). Anything
    looser (">=", "~=", a range, an unpinned name) returns None rather than
    guessing which version within the range is actually installed.
    """
    if not constraint:
        return None
    text = constraint.strip()
    if text.startswith("=="):
        text = text[2:].strip()
    elif not text[0].isdigit() or _LOOSE_CONSTRAINT_RE.search(text):
        return None
    return parse_version(text.split("+")[0])


def dependency_constraint(constraints: Optional[dict], package: str) -> Optional[str]:
    """Look up `package` in a {package: constraint} dict (as
    extract_lockfile/parse_pip_list produce for JobSpec.dependency_constraints),
    normalizing both sides the way pip treats package names - "megatron-core"
    and "megatron_core" are the same package, and a requirements.txt/lockfile
    is free to spell it either way, so a Rule condition can't rely on one
    fixed key spelling.
    """
    if not constraints:
        return None
    normalized = package.lower().replace("_", "-")
    for name, constraint in constraints.items():
        if name.lower().replace("_", "-") == normalized:
            return constraint
    return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_GDR_LEVEL_NAMES = {"LOC", "PIX", "PXB", "PHB", "SYS"}


def parse_gdr_level(value: Any) -> Any:
    """NCCL_NET_GDR_LEVEL (and NCCL_P2P_LEVEL) accept either a named
    distance (LOC/PIX/PXB/PHB/SYS) or a numeric level. The numeric
    meanings have shifted across NCCL versions (SYS was 4 before 2.4.7,
    5 after), so this deliberately does not normalize one form to the
    other - it just parses whichever form was actually given, instead of
    forcing a non-numeric string through int() and losing it to "absent".
    """
    if value is None:
        return None
    text = str(value).strip()
    upper = text.upper()
    if upper in _GDR_LEVEL_NAMES:
        return upper
    return safe_int(text)


def load_yaml_file(path: Path) -> dict:
    """Read and parse a YAML file, tolerating anything that goes wrong -
    missing file, bad YAML, or a document that isn't a mapping - by
    returning an empty dict instead of raising.
    """
    try:
        text = path.read_text()
    except OSError:
        return {}
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return doc if isinstance(doc, dict) else {}
