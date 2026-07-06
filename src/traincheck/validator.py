"""Config parsing and validation logic."""

from dataclasses import dataclass, field
from typing import Any

from traincheck.core import Result, RuleEngine
from traincheck.ir import Field
from traincheck.rules import BUILTIN_RULES
from traincheck.utils import parse_version


def _unset() -> Field:
    """Default for a JobSpec leaf that no parser has populated yet."""
    return Field(value=None, status="absent")

@dataclass
class Meta:
    """Bookkeeping about a JobSpec as a whole - not itself a config value."""

    unresolved: list = field(default_factory=list)

@dataclass
class JobSpec:
    """Flat view of a traincheck config, as consumed by the rule engine.

    Every leaf is a `Field` rather than a bare value, carrying where it came
    from and whether it's actually known.
    """

    # Resources
    nodes: Field = field(default_factory=_unset)
    gpus_per_node: Field = field(default_factory=_unset)
    gpu_type: Field = field(default_factory=_unset)
    interconnect: Field = field(default_factory=_unset)
    gpu_memory_gb: Field = field(default_factory=_unset)
    walltime: Field = field(default_factory=_unset)
    partition: Field = field(default_factory=_unset)
    # Launcher
    world_size: Field = field(default_factory=_unset)
    # Framework / Software
    framework_name: Field = field(default_factory=_unset)
    framework_version: Field = field(default_factory=_unset)
    nccl_version: Field = field(default_factory=_unset)
    nccl_algo: Field = field(default_factory=_unset)
    cuda_version: Field = field(default_factory=_unset)
    # Environment
    nccl_ib_disable: Field = field(default_factory=_unset)
    nccl_net_gdr_level: Field = field(default_factory=_unset)
    # Parallelism
    tensor_parallel: Field = field(default_factory=_unset)
    pipeline_parallel: Field = field(default_factory=_unset)
    data_parallel: Field = field(default_factory=_unset)
    sharding: Field = field(default_factory=_unset)
    # Model / batch
    model_size_billion_params: Field = field(default_factory=_unset)
    train_micro_batch_size_per_gpu: Field = field(default_factory=_unset)
    gradient_accumulation_steps: Field = field(default_factory=_unset)
    # Data
    dataloader_workers: Field = field(default_factory=_unset)
    # Checkpointing
    checkpoint_frequency: Field = field(default_factory=_unset)
    # HostEnv - always live facts, never present in a config file
    driver_version: Field = field(default_factory=_unset)
    kernel_version: Field = field(default_factory=_unset)
    ofed_version: Field = field(default_factory=_unset)
    peermem_loaded: Field = field(default_factory=_unset)

    meta: Meta = field(default_factory=Meta)

def _resolved(value: Any) -> Field:
    """Wrap a value the native parser read straight out of the config."""
    return Field(value=value, status="resolved", source="native", confidence=1.0)

def parse_config(config: dict[str, Any]) -> JobSpec:
    """Parse a traincheck config dictionary into the flat context
    expected by the rule engine.
    """
    nccl = config.get("nccl", {})
    framework = config.get("framework", {})
    parallelism = config.get("parallelism", {})
    cluster = config.get("cluster", {})
    env = config.get("environment", {})

    return JobSpec(
        nodes=_resolved(cluster.get("nodes")),
        gpus_per_node=_resolved(cluster.get("gpus_per_node")),
        gpu_type=_resolved(cluster.get("gpu_type")),
        interconnect=_resolved(cluster.get("interconnect")),
        gpu_memory_gb=_resolved(cluster.get("gpu_memory_gb")),
        walltime=_resolved(cluster.get("walltime")),
        partition=_resolved(cluster.get("partition")),
        world_size=_resolved(None),
        framework_name=_resolved(framework.get("name")),
        framework_version=_resolved(parse_version(framework.get("version"))),
        nccl_version=_resolved(parse_version(nccl.get("version"))),
        nccl_algo=_resolved(nccl.get("algo")),
        cuda_version=_resolved(None),
        nccl_ib_disable=_resolved(env.get("NCCL_IB_DISABLE")),
        nccl_net_gdr_level=_resolved(env.get("NCCL_NET_GDR_LEVEL")),
        tensor_parallel=_resolved(parallelism.get("tensor_parallel")),
        pipeline_parallel=_resolved(parallelism.get("pipeline_parallel")),
        data_parallel=_resolved(parallelism.get("data_parallel")),
        sharding=_resolved(None),
        model_size_billion_params=_resolved(config.get("model", {}).get("size_billion_params")),
        train_micro_batch_size_per_gpu=_resolved(None),
        gradient_accumulation_steps=_resolved(None),
        dataloader_workers=_resolved(config.get("data", {}).get("dataloader_workers")),
        checkpoint_frequency=_resolved(config.get("checkpoint", {}).get("frequency_steps")),
        driver_version=_resolved(None),
        kernel_version=_resolved(None),
        ofed_version=_resolved(None),
        peermem_loaded=_resolved(None),
    )

class Validator:
    def __init__(self):
        self.engine = RuleEngine()
        for rule in BUILTIN_RULES:
            self.engine.register(rule)

    def validate(self, config: dict[str, Any]) -> Result:
        context = parse_config(config)
        return self.engine.check(vars(context))
