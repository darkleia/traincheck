"""Intermediate representation for config-derived values.

Every leaf a parser extracts from a config is wrapped in a `Field` instead of
a bare value, so the rule engine can tell a value it's confident about from
one that still needs a human (or another tool) to confirm.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

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


def build_launcher_fields(launcher: Optional[dict], source: str) -> dict[str, Field]:
    """Turn `extract_shell`'s raw `launcher` dict into the Field-wrapped
    Launcher-section values of a JobSpec.

    Every adapter that runs a script through `extract_shell` needs the same
    elastic-nnodes-range, host-dependent-nproc, torchrun's implicit
    max_restarts default, and standalone/rdzv-endpoint-conflict handling, so
    it belongs here once rather than being reimplemented per adapter.
    """
    launcher = launcher or {}

    nnodes_min = launcher.get("nnodes_min")
    nnodes_max = launcher.get("nnodes_max")
    nproc_per_node = launcher.get("nproc_per_node")
    nproc_host_dependent = launcher.get("nproc_per_node_host_dependent", False)
    is_elastic = nnodes_min is not None and nnodes_min != nnodes_max

    fields = {
        "launcher_kind": resolved_or_absent(launcher.get("kind"), source),
        "launcher_nnodes_min": resolved_or_absent(nnodes_min, source),
        "launcher_nnodes_max": resolved_or_absent(nnodes_max, source),
        "launcher_rdzv_backend": resolved_or_absent(launcher.get("rdzv_backend"), source),
        "launcher_rdzv_endpoint": resolved_or_absent(launcher.get("rdzv_endpoint"), source),
        "launcher_rdzv_id": resolved_or_absent(launcher.get("rdzv_id"), source),
        "launcher_node_rank": resolved_or_absent(launcher.get("node_rank"), source),
        "launcher_master_addr": resolved_or_absent(launcher.get("master_addr"), source),
        "launcher_master_port": resolved_or_absent(launcher.get("master_port"), source),
    }

    if is_elastic:
        fields["launcher_nnodes"] = Field(value=None, status="unknown", reason="elastic node range")
    else:
        fields["launcher_nnodes"] = resolved_or_absent(nnodes_min, source)

    if nproc_host_dependent:
        fields["launcher_nproc_per_node"] = Field(
            value=None, status="unknown", reason="per-node count is host-dependent"
        )
    else:
        fields["launcher_nproc_per_node"] = resolved_or_absent(nproc_per_node, source)

    max_restarts = launcher.get("max_restarts")
    if max_restarts is None:
        fields["launcher_max_restarts"] = Field(value=None, status="absent", source=source)
    elif launcher.get("max_restarts_is_default"):
        fields["launcher_max_restarts"] = Field(
            value=max_restarts,
            status="resolved",
            source=source,
            confidence=1.0,
            reason="torchrun default (no --max-restarts given)",
        )
    else:
        fields["launcher_max_restarts"] = Field(value=max_restarts, status="resolved", source=source, confidence=1.0)

    if launcher.get("standalone"):
        reason = "standalone conflicts with explicit rendezvous endpoint" if launcher.get("standalone_conflict") else ""
        fields["launcher_standalone"] = Field(
            value=True, status="resolved", source=source, confidence=1.0, reason=reason
        )
    else:
        fields["launcher_standalone"] = Field(value=None, status="absent", source=source)

    if is_elastic:
        fields["world_size"] = Field(value=None, status="unknown", reason="elastic node range")
    elif nproc_host_dependent:
        fields["world_size"] = Field(value=None, status="unknown", reason="per-node count is host-dependent")
    elif nnodes_min is not None and isinstance(nproc_per_node, int):
        fields["world_size"] = Field(
            value=nnodes_min * nproc_per_node, status="resolved", source=source, confidence=1.0
        )
    else:
        fields["world_size"] = Field(value=None, status="absent", source=source)

    return fields
