"""Shared shell-body handling for every HPC scheduler adapter (Slurm, PBS,
LSF, SGE).

All four schedulers share the same shape: a directive block (with its own
per-scheduler prefix and flag syntax - handled by each adapter itself) and
then a plain shell body. Once the directives are stripped, everything
downstream - the launcher line, module-load/image versions, comm_env, a
referenced DeepSpeed config, and the always-unknown host-env facts - is
read identically regardless of which scheduler wrote the header, so it
lives here once instead of being reimplemented per scheduler.
"""

import os
from typing import Optional

from traincheck.adapters.deepspeed import adapt_deepspeed
from traincheck.extractors.accelerate import fill_fsdp_sharding
from traincheck.extractors.image import extract_image
from traincheck.extractors.shell import extract_shell
from traincheck.ir import Field, build_comm_env, build_launcher_fields, resolved_or_absent
from traincheck.utils import parse_gdr_level, parse_version, safe_int
from traincheck.validator import JobSpec

HOST_ENV_REASON = "host fact, not in any file"
HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")


def apply_shell_body(spec: JobSpec, body: str, base_dir: str, extra_env: Optional[dict] = None) -> None:
    """Fill in everything a scheduler adapter reads from the shell body
    that remains after its own directive block is stripped out.

    `spec` should already have its scheduler-specific Resources fields
    (nodes, gpus_per_node, gpu_type, walltime, partition/queue) and
    world_size set before this is called - if world_size is already
    resolved from the header, the shell-derived one is dropped rather
    than silently overwriting a more authoritative value.
    """
    shell = extract_shell(body, base_dir=base_dir, extra_env=extra_env)

    launcher_fields = build_launcher_fields(shell["launcher"], "shell")
    if spec.world_size.status == "resolved":
        launcher_fields.pop("world_size")
    for name, launcher_field in launcher_fields.items():
        setattr(spec, name, launcher_field)

    module_loads = shell["module_loads"]
    spec.cuda_version = resolved_or_absent(_module_version(module_loads, "cuda"), "shell")
    spec.nccl_version = resolved_or_absent(parse_version(_module_version(module_loads, "nccl")), "shell")

    env_vars = shell["env_vars"]
    spec.nccl_algo = resolved_or_absent(env_vars.get("NCCL_ALGO"), "shell")
    spec.nccl_ib_disable = resolved_or_absent(safe_int(env_vars.get("NCCL_IB_DISABLE")), "shell")
    spec.nccl_net_gdr_level = resolved_or_absent(parse_gdr_level(env_vars.get("NCCL_NET_GDR_LEVEL")), "shell")

    image_ref = shell["image_ref"]
    image_env = None
    if image_ref:
        image_fields = extract_image(image_ref)
        image_env = image_fields["env"]
        spec.image_pin_status = resolved_or_absent(image_fields["pin_status"], "shell:image")
        if spec.cuda_version.status != "resolved":
            spec.cuda_version = image_fields["cuda"]
        if spec.nccl_version.status != "resolved":
            spec.nccl_version = image_fields["nccl"]
        if spec.framework_version.status != "resolved":
            spec.framework_version = image_fields["framework"]

    # runtime (shell export) takes precedence over image-baked env
    spec.comm_env = build_comm_env([(f"shell:image:{image_ref}", image_env), ("shell", env_vars)])

    framework_config = shell["framework_config"]
    if framework_config is not None:
        ds_config_path = os.path.join(base_dir, framework_config)
        if os.path.isfile(ds_config_path):
            ds_fields = adapt_deepspeed(ds_config_path)
            # Guarded: a Megatron launch flag may have already resolved
            # tensor_parallel/pipeline_parallel/sharding (a real combo in
            # Megatron-DeepSpeed setups) - don't clobber that with "absent"
            # just because the DeepSpeed config doesn't also set it.
            if ds_fields["sharding"].status == "resolved":
                spec.sharding = ds_fields["sharding"]
            if ds_fields["tensor_parallel"].status == "resolved":
                spec.tensor_parallel = ds_fields["tensor_parallel"]
            if ds_fields["pipeline_parallel"].status == "resolved":
                spec.pipeline_parallel = ds_fields["pipeline_parallel"]
            spec.data_parallel = ds_fields["data_parallel"]
            spec.train_micro_batch_size_per_gpu = ds_fields["train_micro_batch_size_per_gpu"]
            spec.gradient_accumulation_steps = ds_fields["gradient_accumulation_steps"]

    fill_fsdp_sharding(spec, shell["launcher"], base_dir)
    derive_data_parallel(spec)

    for name in HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)


def derive_data_parallel(spec: JobSpec) -> None:
    """data_parallel = world_size / (tensor_parallel * pipeline_parallel).

    The DeepSpeed adapter always leaves data_parallel absent, since a
    DeepSpeed config alone never carries world size. By this point,
    world_size and tp/pp (if a DeepSpeed config or Megatron launch flags
    resolved them) may both be known, so it can be derived here - but only
    when tp*pp actually divides world_size evenly. When it doesn't, the
    model-parallel grouping itself is broken (PARALLEL-002 flags exactly
    this), and a floor-divided data_parallel would just be a misleading
    number, not a real replica count.
    """
    if spec.world_size.status != "resolved":
        return
    if spec.tensor_parallel.status != "resolved" or spec.pipeline_parallel.status != "resolved":
        return

    tp = spec.tensor_parallel.value
    pp = spec.pipeline_parallel.value
    if not tp or not pp:
        return
    if spec.world_size.value % (tp * pp) != 0:
        return

    spec.data_parallel = Field(
        value=spec.world_size.value // (tp * pp), status="resolved", source="derived", confidence=1.0
    )


def _module_version(module_loads: list, name: str) -> Optional[str]:
    prefix = f"{name}/"
    for module in module_loads:
        if module.startswith(prefix):
            return module[len(prefix) :]
    return None
