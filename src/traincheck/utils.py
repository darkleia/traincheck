"""Small helpers shared across parsers/adapters."""

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
