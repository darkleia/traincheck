"""Config-coherence rules: internal contradictions within a single resolved
JobSpec (tp*pp != world size, minAvailable < sum(replicas), IB disabled on
an IB cluster, ...).

These are authored directly from the config/scheduler spec, never mined
from the web - there is nothing here for a source URL to confirm, since
the claim is "these two fields, taken from the same job, don't agree with
each other," not "component X version A is incompatible with component Y
version B." Keeping them in their own module means a bad mined
version-incompatibility candidate (see `version_incompat.py`) can never
end up alongside - and corrupt - these.
"""

from traincheck.core import Rule, Severity

CONFIG_COHERENCE_RULES: list[Rule] = [
    Rule(
        id="NCCL-IB-001",
        severity=Severity.ERROR,
        condition=("nccl_ib_disable == 1 and interconnect == 'InfiniBand'"),
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
        fix_suggestion=("Adjust parallelism settings so TP * PP * DP = nodes * gpus_per_node."),
    ),
    Rule(
        id="PARALLEL-002",
        severity=Severity.ERROR,
        condition=(
            "tensor_parallel is not None "
            "and pipeline_parallel is not None "
            "and world_size is not None "
            "and tensor_parallel * pipeline_parallel != 0 "
            "and world_size % (tensor_parallel * pipeline_parallel) != 0"
        ),
        message=(
            "tensor_parallel * pipeline_parallel does not evenly divide world_size - the model-parallel "
            "group size isn't a clean divisor of the total GPU count, so the job can't actually form "
            "complete replica groups."
        ),
        fix_suggestion="Adjust tensor_parallel/pipeline_parallel or world_size so tp * pp evenly divides world_size.",
        detail="f'tensor_parallel={tensor_parallel}, pipeline_parallel={pipeline_parallel}, world_size={world_size}'",
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
        message=("Model with >30B parameters may not fit on GPUs with <80GB memory without tensor parallelism."),
        fix_suggestion=("Enable tensor parallelism (e.g., tensor_parallel=2) or use activation checkpointing."),
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
        message=("DataLoader workers may be insufficient for 8-GPU nodes, causing GPU starvation during data loading."),
        fix_suggestion=("Increase DataLoader workers to at least 4 per GPU, or enable prefetch_factor."),
    ),
    Rule(
        id="CHECKPOINT-001",
        severity=Severity.INFO,
        condition=(
            "checkpoint_frequency is not None and checkpoint_frequency > 1000 and nodes is not None and nodes > 32"
        ),
        message=(
            "Checkpoint frequency is low for a large-scale run. A failure at step 999 would lose significant progress."
        ),
        fix_suggestion=("Consider reducing checkpoint frequency or enabling async checkpointing."),
    ),
    Rule(
        id="GANG-001",
        severity=Severity.WARN,
        condition=(
            "min_available is not None and task_replicas_total is not None and min_available < task_replicas_total"
        ),
        message=(
            "minAvailable is less than the job's total task replicas - the scheduler can admit a partial "
            "gang, and the job will hang waiting for pods that never get scheduled alongside it."
        ),
        fix_suggestion=(
            "Set minAvailable equal to the sum of all task replicas so the whole gang is admitted atomically."
        ),
        detail="f'minAvailable={min_available}, sum(replicas)={task_replicas_total}'",
    ),
    Rule(
        id="GANG-002",
        severity=Severity.WARN,
        condition="scheduler_name == 'volcano' and min_available is None",
        message=(
            "schedulerName is volcano but no PodGroup/minAvailable was found - without gang scheduling, "
            "Volcano may admit only some of the job's pods and the rest will wait indefinitely, deadlocking "
            "collective-communication training."
        ),
        fix_suggestion="Provide a PodGroup (or spec.minAvailable) matching the job's total replica count.",
        detail="f'scheduler_name={scheduler_name!r}'",
    ),
    Rule(
        id="GANG-003",
        severity=Severity.WARN,
        condition="queue_name is not None and scheduler_name != 'volcano'",
        message=(
            "Job is queued via a Kueue queue-name label but has no gang scheduler configured - Kueue admits "
            "the job as a unit but doesn't itself guarantee all its pods start together, so partial "
            "admission is still possible."
        ),
        fix_suggestion="Pair the Kueue queue with a gang-aware scheduler/integration (e.g. Volcano) or a PodGroup.",
        detail="f'queue_name={queue_name!r}, scheduler_name={scheduler_name!r}'",
    ),
]
