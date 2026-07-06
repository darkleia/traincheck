"""Small helpers shared across parsers/adapters."""

from typing import Optional


def parse_version(version: Optional[str]) -> Optional[tuple]:
    """Turn a dotted version string like "2.19" into (2, 19)."""
    if version is None:
        return None
    try:
        return tuple(int(x) for x in version.split("."))
    except (ValueError, AttributeError):
        return None
