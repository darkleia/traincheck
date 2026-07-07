"""Map a DeepSpeed ds_config.json onto the parallelism/model slice of a JobSpec."""

import json
from typing import Any

from traincheck.ir import Field

_TENSOR_PARALLEL_KEYS = ("tensor_parallel_size", "tensor_parallel", "mp_size")
_PIPELINE_PARALLEL_KEYS = ("pipeline_parallel_size", "pipeline_parallel")


def adapt_deepspeed(path: str) -> dict[str, Field]:
    """Read a DeepSpeed config file and return its parallelism/model fields.

    Keys the config doesn't set become status="absent" Fields (genuinely
    not configured), not "unknown" (which is reserved for values we tried
    and failed to resolve).
    """
    source = f"deepspeed:{path}"
    with open(path) as f:
        config: dict[str, Any] = json.load(f)

    zero = config.get("zero_optimization", {})

    return {
        "sharding": _field_for(zero, "stage", source),
        "tensor_parallel": _field_for_any(config, _TENSOR_PARALLEL_KEYS, source),
        "pipeline_parallel": _field_for_any(config, _PIPELINE_PARALLEL_KEYS, source),
        "data_parallel": Field(
            value=None,
            status="absent",
            source=source,
            reason=(
                "data_parallel = world_size / (tensor_parallel * pipeline_parallel); "
                "a DeepSpeed config alone doesn't carry world size, so this can only "
                "be derived once the launcher/scheduler slice provides node and GPU "
                "counts."
            ),
        ),
        "train_micro_batch_size_per_gpu": _field_for(config, "train_micro_batch_size_per_gpu", source),
        "gradient_accumulation_steps": _field_for(config, "gradient_accumulation_steps", source),
    }


_AUTO_REASON = (
    'value is "auto" - DeepSpeed fills this in at runtime from the HF '
    "Trainer/Accelerate's own args, so it can't be read statically"
)


def _field_for(container: dict[str, Any], key: str, source: str) -> Field:
    if key not in container:
        return Field(value=None, status="absent", source=source)
    value = container[key]
    if value == "auto":
        return Field(value=None, status="unknown", reason=_AUTO_REASON)
    return Field(value=value, status="resolved", source=source, confidence=1.0)


def _field_for_any(container: dict[str, Any], keys: tuple, source: str) -> Field:
    for key in keys:
        if key in container:
            value = container[key]
            if value == "auto":
                return Field(value=None, status="unknown", reason=_AUTO_REASON)
            return Field(value=value, status="resolved", source=source, confidence=1.0)
    return Field(value=None, status="absent", source=source)
