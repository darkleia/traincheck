"""Intermediate representation for config-derived values.

Every leaf a parser extracts from a config is wrapped in a `Field` instead of
a bare value, so the rule engine can tell a value it's confident about from
one that still needs a human (or another tool) to confirm.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from traincheck.utils import parse_gdr_level, safe_int

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


# Every NCCL/CUDA comm-related env var comm_env tracks. NCCL_ALGO,
# NCCL_IB_DISABLE and NCCL_NET_GDR_LEVEL also have their own dedicated
# JobSpec fields (kept for existing rules to read); comm_env carries all of
# them plus the rest, uniformly, with provenance.
COMM_ENV_VARS = (
    "NCCL_ALGO",
    "NCCL_IB_DISABLE",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_SOCKET_IFNAME",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_IB_HCA",
    "NCCL_PROTO",
    "NCCL_DEBUG",
    "NCCL_SHM_DISABLE",
    "NCCL_IB_TIMEOUT",
    "NCCL_TIMEOUT",
)

# Accept either a named distance (LOC/PIX/PXB/PHB/SYS) or a numeric level -
# same normalizer already used for NCCL_NET_GDR_LEVEL.
_LEVEL_VARS = {"NCCL_NET_GDR_LEVEL", "NCCL_P2P_LEVEL"}
# 0/1 style switches and small integer counts - normalize to int so a rule
# can compare them numerically instead of string-matching "0" vs 0.
_INT_VARS = {"NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_TIMEOUT", "NCCL_TIMEOUT"}


def build_comm_env(sources: list[tuple[str, Optional[dict[str, Any]]]]) -> dict[str, Field]:
    """Build the comm_env bucket: one Field per var in `COMM_ENV_VARS`,
    read from whichever of `sources` set it.

    `sources` must be given lowest-precedence first (e.g. image-baked env
    before a shell export or k8s container.env) - the last source to set a
    given var wins. When more than one source sets the same var, the
    winning Field's reason records the value it overrode, rather than
    silently dropping it.
    """
    comm_env: dict[str, Field] = {}

    for var in COMM_ENV_VARS:
        seen = [(name, env[var]) for name, env in sources if env and var in env]

        if not seen:
            comm_env[var] = Field(value=None, status="absent")
            continue

        winning_source, winning_raw = seen[-1]
        reason = ""
        if len(seen) > 1:
            other_source, other_raw = seen[-2]
            reason = f"{winning_source} value overrides conflicting value {other_raw!r} from {other_source}"

        comm_env[var] = Field(
            value=_normalize_comm_env_value(var, winning_raw),
            status="resolved",
            source=winning_source,
            confidence=1.0,
            reason=reason,
        )

    return comm_env


def _normalize_comm_env_value(var: str, raw: Any) -> Any:
    if var in _LEVEL_VARS:
        return parse_gdr_level(raw)
    if var in _INT_VARS:
        return safe_int(raw)
    return raw
