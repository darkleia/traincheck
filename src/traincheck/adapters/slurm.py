"""Map a Slurm sbatch script onto a JobSpec.

Pass one reads the #SBATCH header for resource facts (nodes, the full
GPU-request flag matrix, GPU type, walltime, partition). Pass two strips
those directives and runs the remaining shell body through `extract_shell`
for launcher/software facts, merging in a DeepSpeed config's
parallelism/model fields when the launch command points at one. Host-level
facts (driver/kernel/OFED/GPU peermem) can never come from a config file,
so they're always reported unknown and routed to `meta.unresolved` for a
follow-up check.

The header itself only counts up to Slurm's own contiguity rule: once a
real (non-comment, non-blank) command line appears, any #SBATCH after it is
just a shell comment, not a directive - and a heterogeneous job packs
several independent directive sets into one header, separated by a
`#SBATCH hetjob` (or `#SBATCH :`) line. Only the first (primary) het group
is used to resolve the standard single-job fields below; the others are
still parsed (so a het job's header never crashes traincheck) but aren't
merged in, since flattening them would silently conflate two different
groups' requests into one.
"""

import os
import re
from typing import Any, Optional

from traincheck.adapters.deepspeed import adapt_deepspeed
from traincheck.extractors.image import extract_image
from traincheck.extractors.shell import extract_shell
from traincheck.ir import Field, build_comm_env, build_launcher_fields, resolved_or_absent
from traincheck.utils import parse_gdr_level, parse_version, safe_int
from traincheck.validator import JobSpec

# Matches both "--flag=value" and "--flag value" (a bare "--flag" with no
# value at all also matches, group(2) is then None) - Slurm accepts either
# spelling for every directive. Not anchored to end-of-line, so a trailing
# inline "# comment" after the value doesn't break the match.
_SBATCH_DIRECTIVE_RE = re.compile(r"^#SBATCH\s+--([\w-]+)(?:[=\s]+(\S+))?")
_HETJOB_SEPARATOR_RE = re.compile(r"^#SBATCH\s+(hetjob|:)\s*$", re.IGNORECASE)
_SBATCH_LINE_RE = re.compile(r"^\s*#SBATCH\b")

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")

_MUTUAL_EXCLUSIVITY_REASON = "--gpus-per-node and --gres=gpu are mutually exclusive but both are set"


def adapt_slurm(path: str, base_dir: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    directive_groups = _parse_sbatch_directive_groups(text)
    directives = directive_groups[0] if directive_groups else {}
    spec = JobSpec()

    nodes = safe_int(directives.get("nodes"))
    spec.nodes = resolved_or_absent(nodes, "sbatch")

    gpu_request = _resolve_gpu_request(directives, nodes)
    spec.gpus_per_node = Field(
        value=gpu_request["gpus_per_node"],
        status="resolved" if gpu_request["gpus_per_node"] is not None else "absent",
        source="sbatch",
        confidence=1.0,
        reason=gpu_request["gpus_per_node_reason"],
    )
    spec.gpu_type = Field(
        value=gpu_request["gpu_type"],
        status="resolved" if gpu_request["gpu_type"] is not None else "absent",
        source="sbatch",
        confidence=1.0,
        reason=gpu_request["gpu_type_reason"],
    )
    if gpu_request["world_size"] is not None:
        spec.world_size = resolved_or_absent(gpu_request["world_size"], "sbatch")

    spec.walltime = resolved_or_absent(directives.get("time"), "sbatch")
    spec.partition = resolved_or_absent(directives.get("partition"), "sbatch")

    body = _strip_sbatch_lines(text)
    slurm_env = _slurm_runtime_env(directives, nodes, gpu_request["gpus_per_node"])
    shell = extract_shell(body, base_dir=base_dir, extra_env=slurm_env)

    launcher_fields = build_launcher_fields(shell["launcher"], "shell")
    if spec.world_size.status == "resolved":
        # the sbatch header's own GPU request is more authoritative than
        # whatever the launch line implies - keep it, drop the other
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


def _parse_sbatch_directive_groups(text: str) -> list[dict]:
    """Parse the #SBATCH header into one directive dict per heterogeneous
    job group (a single-element list for an ordinary, non-het job).

    Enforces Slurm's own contiguity rule: directives stop being read the
    moment the first real (non-comment, non-blank) command line appears -
    anything under a `#SBATCH` spelling after that point is just a shell
    comment, exactly as Slurm itself treats it.
    """
    groups: list[dict] = [{}]
    seen_command = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            seen_command = True
            continue
        if seen_command:
            continue

        if _HETJOB_SEPARATOR_RE.match(stripped):
            groups.append({})
            continue

        match = _SBATCH_DIRECTIVE_RE.match(stripped)
        if match:
            key, value = match.groups()
            if value is not None:
                groups[-1][key] = value

    return groups


def _strip_sbatch_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _SBATCH_LINE_RE.match(line))


def _module_version(module_loads: list, name: str) -> Optional[str]:
    prefix = f"{name}/"
    for module in module_loads:
        if module.startswith(prefix):
            return module[len(prefix) :]
    return None


def _resolve_gpu_request(directives: dict, nodes: Optional[int]) -> dict[str, Any]:
    """Resolve the full GPU-request flag matrix into (gpus_per_node,
    gpu_type, world_size), plus a `reason` noting any inconsistency found
    along the way (kept alongside the resolved value, not turned into
    "unknown" - a deterministic value was still chosen).

    --gres=gpu[:TYPE]:N and --gpus-per-node[=TYPE]:N are both per-node
    counts; --gpus[=TYPE]:N is a job TOTAL, which used to get conflated
    with a per-node count and fed straight into world_size = count * nodes
    - now it's divided by nodes for gpus_per_node, but the total itself is
    used directly for world_size, since dividing and remultiplying could
    round away from the real total.
    """
    gres_type, gres_count = _parse_gres_gpu(directives.get("gres"))
    gpus_type, gpus_total = _parse_typed_count(directives.get("gpus"))
    per_node_type, per_node_count = _parse_typed_count(directives.get("gpus-per-node"))
    per_task_type, per_task_count = _parse_typed_count(directives.get("gpus-per-task"))

    count_reasons = []
    if per_node_count is not None and gres_count is not None:
        count_reasons.append(_MUTUAL_EXCLUSIVITY_REASON)

    gpus_per_node = None
    if per_node_count is not None:
        gpus_per_node = per_node_count
    elif gres_count is not None:
        gpus_per_node = gres_count
    elif gpus_total is not None and nodes:
        if gpus_total % nodes != 0:
            count_reasons.append(f"--gpus={gpus_total} does not divide evenly across --nodes={nodes}")
        gpus_per_node = gpus_total // nodes
    elif per_task_count is not None:
        ntasks_per_node = safe_int(directives.get("ntasks-per-node"))
        if ntasks_per_node is not None:
            gpus_per_node = per_task_count * ntasks_per_node

    type_reasons = []
    types_seen = {t for t in (gres_type, gpus_type, per_node_type, per_task_type) if t}
    if len(types_seen) > 1:
        type_reasons.append(f"inconsistent GPU type across flags: {sorted(types_seen)}")

    gpu_type = gres_type or gpus_type or per_node_type or per_task_type
    if gpu_type is None and directives.get("constraint"):
        gpu_type = _parse_constraint(directives["constraint"])

    world_size = None
    if gpus_total is not None:
        world_size = gpus_total
    elif gpus_per_node is not None and nodes is not None:
        world_size = gpus_per_node * nodes

    return {
        "gpus_per_node": gpus_per_node,
        "gpus_per_node_reason": "; ".join(count_reasons),
        "gpu_type": gpu_type,
        "gpu_type_reason": "; ".join(type_reasons),
        "world_size": world_size,
    }


def _parse_gres_gpu(value: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """--gres can list multiple resources comma-separated (e.g.
    "gpu:4,license:1"); only the "gpu" entry (if any) matters here.
    """
    if value is None:
        return None, None
    for entry in value.split(","):
        parts = entry.strip().split(":")
        if not parts or parts[0] != "gpu":
            continue
        if len(parts) >= 3:  # gpu:TYPE:N
            return parts[1], safe_int(parts[-1])
        if len(parts) == 2:  # gpu:N
            return None, safe_int(parts[1])
    return None, None


def _parse_typed_count(value: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """--gpus/--gpus-per-node/--gpus-per-task all share a "[TYPE:]N" shape."""
    if value is None:
        return None, None
    parts = value.split(":")
    if len(parts) == 2:
        return parts[0], safe_int(parts[1])
    if len(parts) == 1:
        return None, safe_int(parts[0])
    return None, None


def _parse_constraint(constraint: str) -> Any:
    """--constraint may combine features with & (AND, all required on the
    same node - kept as the single compound expression, not split apart)
    or | (OR, any one suffices - genuinely ambiguous which type a node
    will actually have, so this yields a set of possibilities instead).
    """
    if "|" in constraint:
        return {part.strip() for part in constraint.split("|") if part.strip()}
    return constraint.strip()


def _slurm_runtime_env(directives: dict, nodes: Optional[int], gpus_per_node: Optional[int]) -> dict[str, str]:
    """Slurm-injected env vars a launch line can reference (e.g.
    `--nproc-per-node=$SLURM_GPUS_ON_NODE`) that the script itself never
    exports - computed from the same directives already parsed above.
    """
    env: dict[str, str] = {}
    if gpus_per_node is not None:
        env["SLURM_GPUS_ON_NODE"] = str(gpus_per_node)

    ntasks = safe_int(directives.get("ntasks"))
    if ntasks is None:
        ntasks_per_node = safe_int(directives.get("ntasks-per-node"))
        if ntasks_per_node is not None and nodes is not None:
            ntasks = ntasks_per_node * nodes
    if ntasks is not None:
        env["SLURM_NTASKS"] = str(ntasks)

    return env
