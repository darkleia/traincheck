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
