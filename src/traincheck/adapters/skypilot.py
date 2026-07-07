"""Map a SkyPilot task YAML onto a JobSpec.

Resources (accelerators/num_nodes) and envs come straight off the task
spec. The image and the actual launch command don't have their own
first-class SkyPilot fields, though - `run`/`setup` are just shell text,
so we reuse `extract_shell` on them for the launcher and any referenced
framework/Hydra config, and scan `setup` for a `docker pull` as a
fallback image reference when `resources.image_id` isn't set.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional

from traincheck.extractors.hydra import extract_hydra
from traincheck.extractors.image import extract_image
from traincheck.extractors.shell import extract_shell
from traincheck.ir import Field, build_launcher_fields, resolved_or_absent
from traincheck.utils import load_yaml_file, parse_gdr_level, safe_int
from traincheck.validator import JobSpec

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")

_DOCKER_PULL_RE = re.compile(r"docker\s+pull\s+(\S+)")


def adapt_skypilot(path: str, base_dir: str) -> JobSpec:
    doc = load_yaml_file(Path(path))
    source = "skypilot"

    spec = JobSpec()

    resources = doc.get("resources") or {}
    gpu_type, gpus_per_node = _parse_accelerators(resources.get("accelerators"))
    spec.gpu_type = resolved_or_absent(gpu_type, source)
    spec.gpus_per_node = resolved_or_absent(gpus_per_node, source)

    num_nodes = doc.get("num_nodes")
    spec.nodes = resolved_or_absent(num_nodes, source)
    world_size = num_nodes * gpus_per_node if num_nodes is not None and gpus_per_node is not None else None
    spec.world_size = resolved_or_absent(world_size, source)

    envs = doc.get("envs") or {}
    spec.nccl_algo = resolved_or_absent(envs.get("NCCL_ALGO"), source)
    spec.nccl_ib_disable = resolved_or_absent(safe_int(envs.get("NCCL_IB_DISABLE")), source)
    spec.nccl_net_gdr_level = resolved_or_absent(parse_gdr_level(envs.get("NCCL_NET_GDR_LEVEL")), source)

    setup_text = doc.get("setup") or ""
    run_text = doc.get("run") or ""

    image_ref = _image_ref(resources, setup_text)
    if image_ref:
        image_fields = extract_image(image_ref)
        spec.image_pin_status = resolved_or_absent(image_fields["pin_status"], f"{source}:image")
        spec.cuda_version = image_fields["cuda"]
        spec.nccl_version = image_fields["nccl"]
        spec.framework_version = image_fields["framework"]

    shell = extract_shell(f"{setup_text}\n{run_text}", base_dir=base_dir)
    launcher_fields = build_launcher_fields(shell["launcher"], "shell")
    # num_nodes + accelerators (above) is the more authoritative source for
    # SkyPilot's own world_size - only fall back to the shell-derived one
    # (e.g. an elastic --nnodes range) when that didn't resolve it.
    shell_world_size = launcher_fields.pop("world_size")
    if spec.world_size.status != "resolved":
        spec.world_size = shell_world_size
    for name, launcher_field in launcher_fields.items():
        setattr(spec, name, launcher_field)

    _fill_from_config(spec, shell, base_dir)

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec


def _parse_accelerators(accelerators: Any) -> tuple:
    if isinstance(accelerators, str):
        gpu_type, _, count = accelerators.partition(":")
        return (gpu_type or None), (safe_int(count) if count else None)
    if isinstance(accelerators, dict) and accelerators:
        gpu_type, count = next(iter(accelerators.items()))
        return gpu_type, safe_int(count)
    return None, None


def _image_ref(resources: dict, setup_text: str) -> Optional[str]:
    image_id = resources.get("image_id")
    if image_id:
        return image_id[len("docker:") :] if image_id.startswith("docker:") else image_id

    match = _DOCKER_PULL_RE.search(setup_text)
    return match.group(1) if match else None


def _fill_from_config(spec: JobSpec, shell: dict, base_dir: str) -> None:
    """`extract_shell` finds either a --deepspeed path (framework_config) or
    a --config/--config-name path (config_path); either way, treat it as a
    Hydra root config and try to compose tp/pp/dp/sharding/model out of it.
    """
    config_hint = shell["config_path"] or shell["framework_config"]
    if not config_hint:
        return

    config_full_path = os.path.join(base_dir, config_hint)
    if not os.path.isfile(config_full_path):
        return

    hydra_fields = extract_hydra(config_full_path, overrides=shell["config_overrides"])
    source = f"hydra:{config_full_path}"

    spec.tensor_parallel = resolved_or_absent(hydra_fields.get("tensor_parallel"), source)
    spec.pipeline_parallel = resolved_or_absent(hydra_fields.get("pipeline_parallel"), source)
    spec.data_parallel = resolved_or_absent(hydra_fields.get("data_parallel"), source)
    spec.sharding = resolved_or_absent(hydra_fields.get("sharding"), source)

    model = hydra_fields.get("model") or {}
    spec.model_size_billion_params = resolved_or_absent(model.get("size_billion_params"), source)
