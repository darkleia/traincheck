"""Map a bare-metal (no-scheduler) launch script onto a JobSpec.

There's no scheduler here at all - no #SBATCH/#PBS/#BSUB/#$ - so nothing
tells us what hardware this script actually runs on. Every Resources
field (node/GPU counts, GPU type, interconnect, walltime, partition) is
reported unknown rather than guessed at; that's a fundamentally different
gap from "absent" (a source that was checked and simply didn't set it).

What the script *does* know about itself - the launcher command's own
--nnodes/--nproc-per-node, env vars, a referenced DeepSpeed config, a
referenced Hydra config, an image - all resolve normally via the same
extract_shell/adapt_deepspeed/extract_hydra/extract_image pieces the
other adapters use.
"""

import os

from traincheck.adapters.deepspeed import adapt_deepspeed
from traincheck.adapters.hpc_shell import derive_data_parallel
from traincheck.extractors.accelerate import fill_fsdp_sharding
from traincheck.extractors.hydra import extract_hydra
from traincheck.extractors.image import extract_image
from traincheck.extractors.shell import extract_shell
from traincheck.ir import Field, build_comm_env, build_launcher_fields, resolved_or_absent
from traincheck.utils import parse_gdr_level, safe_int
from traincheck.validator import JobSpec

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")

_NO_SCHEDULER_REASON = "no scheduler in entrypoint"
_RESOURCE_FIELDS = (
    "nodes",
    "gpus_per_node",
    "gpu_type",
    "interconnect",
    "gpu_memory_gb",
    "walltime",
    "partition",
)


def adapt_bare(path: str, base_dir: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    spec = JobSpec()

    for name in _RESOURCE_FIELDS:
        setattr(spec, name, Field(value=None, status="unknown", reason=_NO_SCHEDULER_REASON))

    shell = extract_shell(text, base_dir=base_dir)
    source = "shell"

    for name, launcher_field in build_launcher_fields(shell["launcher"], source).items():
        setattr(spec, name, launcher_field)

    env_vars = shell["env_vars"]
    spec.nccl_algo = resolved_or_absent(env_vars.get("NCCL_ALGO"), source)
    spec.nccl_ib_disable = resolved_or_absent(safe_int(env_vars.get("NCCL_IB_DISABLE")), source)
    spec.nccl_net_gdr_level = resolved_or_absent(parse_gdr_level(env_vars.get("NCCL_NET_GDR_LEVEL")), source)

    image_ref = shell["image_ref"]
    image_env = None
    if image_ref:
        image_fields = extract_image(image_ref)
        image_env = image_fields["env"]
        spec.image_pin_status = resolved_or_absent(image_fields["pin_status"], f"{source}:image")
        spec.cuda_version = image_fields["cuda"]
        spec.nccl_version = image_fields["nccl"]
        spec.framework_version = image_fields["framework"]

    # runtime (shell export) takes precedence over image-baked env
    spec.comm_env = build_comm_env([(f"{source}:image:{image_ref}", image_env), (source, env_vars)])

    _fill_deepspeed(spec, shell, base_dir)
    _fill_hydra(spec, shell, base_dir)
    fill_fsdp_sharding(spec, shell["launcher"], base_dir)
    derive_data_parallel(spec)

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec


def _fill_deepspeed(spec: JobSpec, shell: dict, base_dir: str) -> None:
    framework_config = shell["framework_config"]
    if framework_config is None:
        return

    ds_config_path = os.path.join(base_dir, framework_config)
    if not os.path.isfile(ds_config_path):
        return

    ds_fields = adapt_deepspeed(ds_config_path)
    # Guarded: a Megatron launch flag may have already resolved
    # tensor_parallel/pipeline_parallel/sharding (a real combo in
    # Megatron-DeepSpeed setups) - don't clobber that with "absent" just
    # because the DeepSpeed config doesn't also set it.
    if ds_fields["sharding"].status == "resolved":
        spec.sharding = ds_fields["sharding"]
    if ds_fields["tensor_parallel"].status == "resolved":
        spec.tensor_parallel = ds_fields["tensor_parallel"]
    if ds_fields["pipeline_parallel"].status == "resolved":
        spec.pipeline_parallel = ds_fields["pipeline_parallel"]
    spec.data_parallel = ds_fields["data_parallel"]
    spec.train_micro_batch_size_per_gpu = ds_fields["train_micro_batch_size_per_gpu"]
    spec.gradient_accumulation_steps = ds_fields["gradient_accumulation_steps"]


def _fill_hydra(spec: JobSpec, shell: dict, base_dir: str) -> None:
    """extract_shell's `config_path` is the --config/--config-name flag -
    a distinct signal from --deepspeed's framework_config, so this never
    competes with (or clobbers) _fill_deepspeed above.
    """
    config_path_hint = shell["config_path"]
    if not config_path_hint:
        return

    hydra_config_path = os.path.join(base_dir, config_path_hint)
    if not os.path.isfile(hydra_config_path):
        return

    hydra_fields = extract_hydra(hydra_config_path, overrides=shell["config_overrides"])
    source = f"hydra:{hydra_config_path}"

    if hydra_fields.get("tensor_parallel") is not None:
        spec.tensor_parallel = resolved_or_absent(hydra_fields["tensor_parallel"], source)
    if hydra_fields.get("pipeline_parallel") is not None:
        spec.pipeline_parallel = resolved_or_absent(hydra_fields["pipeline_parallel"], source)
    if hydra_fields.get("data_parallel") is not None:
        spec.data_parallel = resolved_or_absent(hydra_fields["data_parallel"], source)
    if hydra_fields.get("sharding") is not None:
        spec.sharding = resolved_or_absent(hydra_fields["sharding"], source)

    model = hydra_fields.get("model") or {}
    if model.get("size_billion_params") is not None:
        spec.model_size_billion_params = resolved_or_absent(model["size_billion_params"], source)
