"""Map a Slurm sbatch script onto a JobSpec.

Pass one reads the #SBATCH header for resource facts (nodes, GPUs per node,
GPU type, walltime, partition). Pass two strips those directives and runs
the remaining shell body through `extract_shell` for launcher/software
facts, merging in a DeepSpeed config's parallelism/model fields when the
launch command points at one. Host-level facts (driver/kernel/OFED/GPU
peermem) can never come from a config file, so they're always reported
unknown and routed to `meta.unresolved` for a follow-up check.
"""

import os
import re
from typing import Optional

from traincheck.adapters.deepspeed import adapt_deepspeed
from traincheck.extractors.shell import extract_shell
from traincheck.ir import Field, resolved_or_absent
from traincheck.utils import parse_version, safe_int
from traincheck.validator import JobSpec

_SBATCH_DIRECTIVE_RE = re.compile(r"^\s*#SBATCH\s+--([\w-]+)=(\S+)\s*$")
_SBATCH_LINE_RE = re.compile(r"^\s*#SBATCH\b")

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")


def adapt_slurm(path: str, base_dir: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    directives = _parse_sbatch_directives(text)
    spec = JobSpec()

    spec.nodes = resolved_or_absent(safe_int(directives.get("nodes")), "sbatch")
    spec.gpus_per_node = resolved_or_absent(safe_int(directives.get("gpus-per-node")), "sbatch")
    spec.gpu_type = resolved_or_absent(directives.get("constraint"), "sbatch")
    spec.walltime = resolved_or_absent(directives.get("time"), "sbatch")
    spec.partition = resolved_or_absent(directives.get("partition"), "sbatch")

    body = _strip_sbatch_lines(text)
    shell = extract_shell(body, base_dir=base_dir)

    launcher = shell["launcher"] or {}
    nnodes = launcher.get("nnodes")
    nproc_per_node = launcher.get("nproc_per_node")
    world_size = nnodes * nproc_per_node if nnodes is not None and nproc_per_node is not None else None
    spec.world_size = resolved_or_absent(world_size, "shell")
    spec.launcher_nproc_per_node = resolved_or_absent(nproc_per_node, "shell")

    module_loads = shell["module_loads"]
    spec.cuda_version = resolved_or_absent(_module_version(module_loads, "cuda"), "shell")
    spec.nccl_version = resolved_or_absent(
        parse_version(_module_version(module_loads, "nccl")), "shell"
    )

    env_vars = shell["env_vars"]
    spec.nccl_algo = resolved_or_absent(env_vars.get("NCCL_ALGO"), "shell")
    spec.nccl_ib_disable = resolved_or_absent(safe_int(env_vars.get("NCCL_IB_DISABLE")), "shell")
    spec.nccl_net_gdr_level = resolved_or_absent(
        safe_int(env_vars.get("NCCL_NET_GDR_LEVEL")), "shell"
    )

    framework_config = shell["framework_config"]
    if framework_config is not None:
        ds_config_path = os.path.join(base_dir, framework_config)
        ds_fields = adapt_deepspeed(ds_config_path)
        spec.sharding = ds_fields["sharding"]
        spec.tensor_parallel = ds_fields["tensor_parallel"]
        spec.pipeline_parallel = ds_fields["pipeline_parallel"]
        spec.data_parallel = ds_fields["data_parallel"]
        spec.train_micro_batch_size_per_gpu = ds_fields["train_micro_batch_size_per_gpu"]
        spec.gradient_accumulation_steps = ds_fields["gradient_accumulation_steps"]

    _derive_data_parallel(spec)

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec


def _derive_data_parallel(spec: JobSpec) -> None:
    """data_parallel = world_size / (tensor_parallel * pipeline_parallel).

    The DeepSpeed adapter always leaves data_parallel absent, since a
    DeepSpeed config alone never carries world size. By this point in the
    Slurm+shell pipeline, world_size and tp/pp (if a DeepSpeed config was
    merged in) may both be resolved, so we can derive it here.
    """
    if spec.world_size.status != "resolved":
        return
    if spec.tensor_parallel.status != "resolved" or spec.pipeline_parallel.status != "resolved":
        return

    tp = spec.tensor_parallel.value
    pp = spec.pipeline_parallel.value
    if not tp or not pp:
        return

    spec.data_parallel = Field(
        value=spec.world_size.value // (tp * pp), status="resolved", source="derived", confidence=1.0
    )


def _parse_sbatch_directives(text: str) -> dict:
    directives = {}
    for line in text.splitlines():
        match = _SBATCH_DIRECTIVE_RE.match(line)
        if match:
            key, value = match.groups()
            directives[key] = value
    return directives


def _strip_sbatch_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _SBATCH_LINE_RE.match(line))


def _module_version(module_loads: list, name: str) -> Optional[str]:
    prefix = f"{name}/"
    for module in module_loads:
        if module.startswith(prefix):
            return module[len(prefix):]
    return None
