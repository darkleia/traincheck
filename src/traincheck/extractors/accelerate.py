"""Read an Accelerate launch config (`default_config.yaml`) and map its
settings onto a JobSpec.

Two entrypoints share this module's mapping logic:
- `adapters/accelerate.py` calls it directly when the config file itself
  is what traincheck was pointed at.
- `adapters/hpc_shell.py`/`adapters/bare.py` call `apply_accelerate_launch`
  when a shell script instead runs `accelerate launch --config_file ...`,
  where the launch line's own flags (--num_processes/--num_machines/
  --mixed_precision) override whatever the file itself says - the file is
  just the starting point a launch flag may deviate from, same as any
  other launcher.

Accelerate frequently carries the whole parallelism story itself, either
directly (num_processes/num_machines) or through an embedded DeepSpeed/
FSDP block (`deepspeed_config`/`fsdp_config`) - each routed through the
existing DeepSpeed/FSDP mapping rather than reimplemented here. DeepSpeed's
own ds_config.json shape is nested (`zero_optimization.stage`,
`zero_optimization.offload_optimizer.device`); Accelerate's config
flattens the handful of DeepSpeed settings its wizard exposes
(`zero_stage`, `offload_optimizer_device`, ...), so that block is
translated into the nested shape first to reuse `map_deepspeed_config`.
"""

import os
from pathlib import Path
from typing import Any, Optional

from traincheck.adapters.deepspeed import map_deepspeed_config
from traincheck.ir import Field, resolved_or_absent
from traincheck.utils import load_yaml_file, safe_int
from traincheck.validator import JobSpec

_FSDP_SHARDING_STRATEGIES = ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2")
_FSDP_SHARDING_STRATEGY_BY_INDEX = dict(enumerate(_FSDP_SHARDING_STRATEGIES, start=1))

_FSDP_ONLY_IN_PYTHON_REASON = (
    "FSDP sharding strategy isn't in a readable Accelerate config - it's either configured directly in "
    "Python (FullyShardedDataParallel(...)) or via Accelerate's own default config, neither of which "
    "can be read statically"
)

_CONFIG_KEYS = (
    "compute_environment",
    "distributed_type",
    "num_processes",
    "num_machines",
    "machine_rank",
    "main_process_ip",
    "main_process_port",
    "mixed_precision",
    "gpu_ids",
)


def extract_accelerate_config(path: str) -> dict[str, Any]:
    """Read an Accelerate launch config YAML into its top-level settings
    plus its raw (unmapped) deepspeed_config/fsdp_config sub-blocks.
    """
    doc = load_yaml_file(Path(path))
    fields: dict[str, Any] = {key: doc.get(key) for key in _CONFIG_KEYS}
    fields["deepspeed_config"] = doc.get("deepspeed_config")
    fields["fsdp_config"] = doc.get("fsdp_config")
    return fields


def apply_accelerate_config(
    spec: JobSpec, fields: dict[str, Any], source: str, overrides: Optional[dict[str, Any]] = None
) -> None:
    """Map an Accelerate config's top-level settings onto the standard
    Launcher fields (world_size/nnodes/node_rank/master_addr/master_port -
    Accelerate's own names for exactly these same concepts), plus
    compute_environment/distributed_type/mixed_precision/gpu_ids.

    world_size is only set here if it isn't already resolved - a script's
    scheduler header (e.g. Slurm's own GPU-request flags) is the more
    authoritative source when both are present, same precedent as every
    other launcher.
    """
    overrides = overrides or {}

    def value(key: str) -> Any:
        override = overrides.get(key)
        return override if override is not None else fields.get(key)

    num_processes = safe_int(value("num_processes"))
    num_machines = safe_int(value("num_machines"))

    spec.launcher_kind = resolved_or_absent("accelerate", source)
    if spec.world_size.status != "resolved":
        spec.world_size = resolved_or_absent(num_processes, source)
    spec.launcher_nnodes = resolved_or_absent(num_machines, source)
    spec.launcher_nnodes_min = resolved_or_absent(num_machines, source)
    spec.launcher_nnodes_max = resolved_or_absent(num_machines, source)
    spec.launcher_node_rank = resolved_or_absent(safe_int(fields.get("machine_rank")), source)
    spec.launcher_master_addr = resolved_or_absent(fields.get("main_process_ip"), source)
    spec.launcher_master_port = resolved_or_absent(safe_int(fields.get("main_process_port")), source)

    if num_processes is not None and num_machines and num_processes % num_machines == 0:
        spec.launcher_nproc_per_node = resolved_or_absent(num_processes // num_machines, source)

    spec.compute_environment = resolved_or_absent(fields.get("compute_environment"), source)
    spec.distributed_type = resolved_or_absent(fields.get("distributed_type"), source)
    spec.mixed_precision = resolved_or_absent(value("mixed_precision"), source)
    spec.gpu_ids = resolved_or_absent(fields.get("gpu_ids"), source)


def route_embedded_frameworks(spec: JobSpec, fields: dict[str, Any], source: str) -> None:
    """DeepSpeed and FSDP settings can be embedded directly in an
    Accelerate config instead of a separate file - route each through the
    same mapping the standalone DeepSpeed adapter / FSDP normalizer use.

    Each destination field is only overwritten when this config actually
    resolves it, so an already-resolved value (e.g. from a Megatron launch
    flag) isn't clobbered back to absent.
    """
    deepspeed_config = fields.get("deepspeed_config")
    if deepspeed_config:
        ds_fields = map_deepspeed_config(_translate_accelerate_deepspeed(deepspeed_config), f"{source}:deepspeed")
        for name in (
            "sharding",
            "zero_offload",
            "tensor_parallel",
            "pipeline_parallel",
            "train_micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
        ):
            if ds_fields[name].status == "resolved":
                setattr(spec, name, ds_fields[name])
        return

    fsdp_config = fields.get("fsdp_config")
    if fsdp_config:
        sharding = _normalize_sharding_strategy(fsdp_config.get("fsdp_sharding_strategy"))
        if sharding is not None:
            spec.sharding = resolved_or_absent(sharding, f"{source}:fsdp")
        elif fields.get("distributed_type") == "FSDP":
            # a real FSDP config that, unusually, doesn't set its own strategy
            spec.sharding = Field(value=None, status="unknown", reason=_FSDP_ONLY_IN_PYTHON_REASON)


def apply_accelerate_launch(spec: JobSpec, launcher: Optional[dict], base_dir: str) -> None:
    """Handle an `accelerate launch` line found in a shell body: read its
    --config_file (if any) and apply it - with the launch line's own
    --num_processes/--num_machines/--mixed_precision overriding the file -
    then route any embedded DeepSpeed/FSDP block. A no-op unless the
    launcher actually is accelerate.
    """
    launcher = launcher or {}
    if launcher.get("kind") != "accelerate":
        return

    source = "accelerate"
    overrides = {
        "num_processes": launcher.get("num_processes"),
        "num_machines": launcher.get("num_machines"),
        "mixed_precision": launcher.get("mixed_precision"),
    }

    config_hint = launcher.get("accelerate_config")
    config_path = os.path.join(base_dir, config_hint) if config_hint else None

    if config_path is not None and os.path.isfile(config_path):
        config_source = f"{source}:{config_path}"
        fields = extract_accelerate_config(config_path)
        apply_accelerate_config(spec, fields, config_source, overrides=overrides)
        route_embedded_frameworks(spec, fields, config_source)
        return

    apply_accelerate_config(spec, {}, source, overrides=overrides)
    if spec.sharding.status != "resolved":
        # no config to confirm FSDP either way - see module docstring
        spec.sharding = Field(value=None, status="unknown", reason=_FSDP_ONLY_IN_PYTHON_REASON)


def _translate_accelerate_deepspeed(block: dict[str, Any]) -> dict[str, Any]:
    """Accelerate's config flattens the handful of DeepSpeed settings its
    wizard exposes instead of using DeepSpeed's own nested ds_config.json
    shape - translate just enough of it to reuse DeepSpeed's own mapping.
    """
    zero_optimization: dict[str, Any] = {}
    if block.get("zero_stage") is not None:
        zero_optimization["stage"] = block["zero_stage"]

    optimizer_device = block.get("offload_optimizer_device")
    if optimizer_device not in (None, "none"):
        zero_optimization["offload_optimizer"] = {"device": optimizer_device}

    param_device = block.get("offload_param_device")
    if param_device not in (None, "none"):
        zero_optimization["offload_param"] = {"device": param_device}

    return {
        "zero_optimization": zero_optimization,
        "gradient_accumulation_steps": block.get("gradient_accumulation_steps"),
    }


def _normalize_sharding_strategy(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return _FSDP_SHARDING_STRATEGY_BY_INDEX.get(raw)
    return str(raw).strip().upper()
