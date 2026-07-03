"""Config parsing and validation logic."""

from typing import Any, Dict, Optional

from traincheck.core import Result, RuleEngine
from traincheck.rules import BUILTIN_RULES

def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a traincheck config dictionary into the flat context
    expected by the rule engine.
    """
    nccl = config.get("nccl", {})
    framework = config.get("framework", {})
    parallelism = config.get("parallelism", {})
    cluster = config.get("cluster", {})
    env = config.get("environment", {})

    return {
        # Cluster
        "nodes": cluster.get("nodes"),
        "gpus_per_node": cluster.get("gpus_per_node"),
        "gpu_type": cluster.get("gpu_type"),
        "interconnect": cluster.get("interconnect"),
        "gpu_memory_gb": cluster.get("gpu_memory_gb"),
        # Framework
        "framework_name": framework.get("name"),
        "framework_version": _parse_version(framework.get("version")),
        # NCCL
        "nccl_version": _parse_version(nccl.get("version")),
        "nccl_algo": nccl.get("algo"),
        # Environment
        "nccl_ib_disable": env.get("NCCL_IB_DISABLE"),
        "nccl_net_gdr_level": env.get("NCCL_NET_GDR_LEVEL"),
        # Parallelism
        "tensor_parallel": parallelism.get("tensor_parallel"),
        "pipeline_parallel": parallelism.get("pipeline_parallel"),
        "data_parallel": parallelism.get("data_parallel"),
        # Model
        "model_size_billion_params": config.get("model", {}).get("size_billion_params"),
        # Data
        "dataloader_workers": config.get("data", {}).get("dataloader_workers"),
        # Checkpointing
        "checkpoint_frequency": config.get("checkpoint", {}).get("frequency_steps"),
    }

def _parse_version(version: Optional[str]) -> Optional[tuple]:
    if version is None:
        return None
    try:
        return tuple(int(x) for x in version.split("."))
    except (ValueError, AttributeError):
        return None

class Validator:
    def __init__(self):
        self.engine = RuleEngine()
        for rule in BUILTIN_RULES:
            self.engine.register(rule)

    def validate(self, config: Dict[str, Any]) -> Result:
        context = parse_config(config)
        return self.engine.check(context)