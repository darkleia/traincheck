"""Read FSDP's sharding strategy out of an Accelerate launch config.

Accelerate itself doesn't take a --fsdp-sharding-strategy flag on the
command line - FSDP is configured entirely through the YAML file
`accelerate launch --config_file <path>` points at (or, if that flag is
omitted, Accelerate's own default config, or FSDP set up directly in
Python via `FullyShardedDataParallel(...)` - neither of which is visible
to a static read at all).
"""

import os
from pathlib import Path
from typing import Any, Optional

from traincheck.ir import Field, resolved_or_absent
from traincheck.utils import load_yaml_file
from traincheck.validator import JobSpec

# Accelerate's own accepted strategy names. Older Accelerate versions
# stored the equivalent torch.distributed.fsdp.ShardingStrategy enum's
# integer value instead of the name - both forms are normalized here.
_FSDP_SHARDING_STRATEGIES = ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2")
_FSDP_SHARDING_STRATEGY_BY_INDEX = dict(enumerate(_FSDP_SHARDING_STRATEGIES, start=1))

_FSDP_ONLY_IN_PYTHON_REASON = (
    "FSDP sharding strategy isn't in a readable Accelerate config - it's either configured directly in "
    "Python (FullyShardedDataParallel(...)) or via Accelerate's own default config, neither of which "
    "can be read statically"
)


def extract_accelerate_config(path: str) -> dict[str, Any]:
    """Read an Accelerate launch config YAML and return its distributed
    type and, when using FSDP, its sharding strategy (normalized to one of
    `_FSDP_SHARDING_STRATEGIES`).
    """
    doc = load_yaml_file(Path(path))
    fsdp_config = doc.get("fsdp_config") or {}
    return {
        "distributed_type": doc.get("distributed_type"),
        "sharding": _normalize_sharding_strategy(fsdp_config.get("fsdp_sharding_strategy")),
    }


def _normalize_sharding_strategy(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return _FSDP_SHARDING_STRATEGY_BY_INDEX.get(raw)
    return str(raw).strip().upper()


def fill_fsdp_sharding(spec: JobSpec, launcher: Optional[dict], base_dir: str) -> None:
    """Populate spec.sharding from an Accelerate --config_file's FSDP
    settings, when the launch line is an `accelerate launch` invocation.

    Only overwrites `sharding` when it isn't already resolved (a DeepSpeed
    ZeRO stage or Megatron-adjacent source may have already set it, and
    those aren't mutually exclusive with a bare `accelerate` launch line -
    e.g. accelerate wrapping a DeepSpeed-configured run).
    """
    launcher = launcher or {}
    if launcher.get("kind") != "accelerate" or spec.sharding.status == "resolved":
        return

    config_hint = launcher.get("accelerate_config")
    if not config_hint:
        spec.sharding = Field(value=None, status="unknown", reason=_FSDP_ONLY_IN_PYTHON_REASON)
        return

    config_path = os.path.join(base_dir, config_hint)
    if not os.path.isfile(config_path):
        spec.sharding = Field(value=None, status="unknown", reason=_FSDP_ONLY_IN_PYTHON_REASON)
        return

    fields = extract_accelerate_config(config_path)
    if fields.get("sharding") is not None:
        spec.sharding = resolved_or_absent(fields["sharding"], f"accelerate:{config_path}")
    elif fields.get("distributed_type") == "FSDP":
        # a real FSDP config that, unusually, doesn't set its own strategy
        spec.sharding = Field(value=None, status="unknown", reason=_FSDP_ONLY_IN_PYTHON_REASON)
    # else: this config plainly isn't using FSDP at all - nothing to flag
