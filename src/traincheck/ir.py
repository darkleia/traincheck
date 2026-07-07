"""Intermediate representation for config-derived values.

Every leaf a parser extracts from a config is wrapped in a `Field` instead of
a bare value, so the rule engine can tell a value it's confident about from
one that still needs a human (or another tool) to confirm.
"""

from dataclasses import dataclass
from typing import Any, Literal

FieldStatus = Literal["resolved", "absent", "unknown"]


@dataclass
class Field:
    value: Any
    status: FieldStatus
    source: str = ""
    confidence: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status == "unknown" and not self.reason:
            raise ValueError("Field(status='unknown') requires a non-empty reason")


def resolved_or_absent(value: Any, source: str = "", confidence: float = 1.0) -> Field:
    """Wrap a value a parser looked for: absent if it wasn't there, resolved
    (at the given confidence) otherwise. Never "unknown" - that status is
    reserved for values a parser tried and failed to determine, not ones
    that were simply never set.
    """
    if value is None:
        return Field(value=None, status="absent", source=source)
    return Field(value=value, status="resolved", source=source, confidence=confidence)
