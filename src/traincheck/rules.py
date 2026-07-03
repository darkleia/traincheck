"""Built-in rules for common GPU training misconfigurations."""

from traincheck.core import Rule, Severity

BUILTIN_RULES: list[Rule] = [
    Rule(
        id="NCCL-RING-001",
        severity=Severity.ERROR,
        condition=(
            "nccl_algo == 'Ring' "
            "and nodes > 32 "
            "and gpu_type in ('A100', 'A100-SXM4-80GB') "
            "and nccl_version < (2, 21)"
        ),
        message=(
            "NCCL Ring algorithm on A100 clusters with >32 nodes has a known "
            "deadlock risk with NCCL versions prior to 2.21."
        ),
        fix_suggestion="Upgrade NCCL to >= 2.21.5 or set nccl_algo='Tree'.",
    ),
    Rule(
        id="NCCL-IB-001",
        severity=Severity.ERROR,
        condition=(
            "nccl_ib_disable == 1 "
            "and interconnect == 'InfiniBand'"
        ),
        message=(
            "InfiniBand is disabled via NCCL_IB_DISABLE=1 but the cluster "
            "interconnect is InfiniBand. This will force communication over "
            "slow sockets and cripple training throughput."
        ),
        fix_suggestion="Set NCCL_IB_DISABLE=0.",
    ),
    Rule(
        id="NCCL-GDR-001",
        severity=Severity.WARN,
        condition=(
            "nccl_net_gdr_level is not None "
            "and nccl_net_gdr_level < 5 "
            "and gpu_type is not None "
            "and str(gpu_type).startswith('H100')"
        ),
        message=(
            "NCCL_NET_GDR_LEVEL below 5 on H100 GPUs disables direct RDMA "
            "between GPU and NIC, significantly reducing communication bandwidth."
        ),
        fix_suggestion="Set NCCL_NET_GDR_LEVEL=5 for H100 clusters.",
    ),
    Rule(
        id="PARALLEL-001",
        severity=Severity.ERROR,
        condition=(
            "tensor_parallel is not None "
            "and pipeline_parallel is not None "
            "and data_parallel is not None "
            "and nodes is not None "
            "and gpus_per_node is not None "
            "and tensor_parallel * pipeline_parallel * data_parallel "
            "!= nodes * gpus_per_node"
        ),
        message=(
            "The product of tensor_parallel, pipeline_parallel, and "
            "data_parallel must equal the total number of GPUs "
            "(nodes * gpus_per_node)."
        ),
        fix_suggestion=(
            "Adjust parallelism settings so TP * PP * DP = "
            "nodes * gpus_per_node."
        ),
    ),
    Rule(
        id="MEMORY-001",
        severity=Severity.WARN,
        condition=(
            "gpu_memory_gb is not None "
            "and model_size_billion_params is not None "
            "and gpu_memory_gb < 80 "
            "and model_size_billion_params > 30 "
            "and tensor_parallel == 1"
        ),
        message=(
            "Model with >30B parameters may not fit on GPUs with <80GB memory "
            "without tensor parallelism."
        ),
        fix_suggestion=(
            "Enable tensor parallelism (e.g., tensor_parallel=2) "
            "or use activation checkpointing."
        ),
    ),
    Rule(
        id="DATALOADER-001",
        severity=Severity.WARN,
        condition=(
            "dataloader_workers is not None "
            "and dataloader_workers < 4 "
            "and gpus_per_node is not None "
            "and gpus_per_node >= 8"
        ),
        message=(
            "DataLoader workers may be insufficient for 8-GPU nodes, "
            "causing GPU starvation during data loading."
        ),
        fix_suggestion=(
            "Increase DataLoader workers to at least 4 per GPU, "
            "or enable prefetch_factor."
        ),
    ),
    Rule(
        id="CHECKPOINT-001",
        severity=Severity.INFO,
        condition=(
            "checkpoint_frequency is not None "
            "and checkpoint_frequency > 1000 "
            "and nodes is not None "
            "and nodes > 32"
        ),
        message=(
            "Checkpoint frequency is low for a large-scale run. "
            "A failure at step 999 would lose significant progress."
        ),
        fix_suggestion=(
            "Consider reducing checkpoint frequency or enabling "
            "async checkpointing."
        ),
    ),
]