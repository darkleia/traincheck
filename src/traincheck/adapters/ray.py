"""Map a Ray cluster config and/or Train job script onto a JobSpec.

Ray's config is split across two very different files: `cluster.yaml`
(the autoscaler config - node types, GPU counts, the docker image) and a
plain Python job script (a `runtime_env` dict passed to `ray.init()`, and
`@ray.remote(num_gpus=...)` task/actor decorators). `adapt_ray` accepts
either one as `path` and looks in `base_dir` for its counterpart, so the
result is the same either way.

There's deliberately no srun/torchrun here - Ray doesn't use one, so
`launcher_kind` is always the literal "ray" rather than something read
out of a shell command. And since the job-script signals are read via
`ast`, not executed, anything built dynamically (a runtime_env assembled
by a function call, a `num_gpus=` expression rather than a literal) is
reported unknown rather than silently skipped - it's a real value we
just can't read without running the code.
"""

import ast
from pathlib import Path
from typing import Any, Optional

from traincheck.extractors.image import extract_image
from traincheck.extractors.lockfile import parse_pip_list
from traincheck.ir import Field, build_comm_env, resolved_or_absent
from traincheck.utils import load_yaml_file, parse_gdr_level, safe_int
from traincheck.validator import JobSpec

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")

_DYNAMIC_RUNTIME_ENV_REASON = (
    "runtime_env is built dynamically in the script (not a literal dict), "
    "so pip/env_vars can't be read without executing the code"
)
_DYNAMIC_NUM_GPUS_REASON = (
    "num_gpus on @ray.remote is a dynamic expression, not a literal, so it can't be read without executing the code"
)


def adapt_ray(path: str, base_dir: str) -> JobSpec:
    p = Path(path)
    base = Path(base_dir)
    source = "ray"

    if p.suffix in (".yaml", ".yml"):
        cluster_doc = load_yaml_file(p)
        job_path = _find_job_py(base)
        job_text = job_path.read_text() if job_path else None
    else:
        job_text = p.read_text()
        cluster_path = _find_cluster_yaml(base)
        cluster_doc = load_yaml_file(cluster_path) if cluster_path else {}

    spec = JobSpec()
    spec.launcher_kind = resolved_or_absent("ray", source)

    image_env, image_ref = (None, None)
    if cluster_doc:
        image_env, image_ref = _fill_from_cluster(spec, cluster_doc, source)

    job_env_vars = None
    if job_text is not None:
        job_env_vars = _fill_from_job_py(spec, job_text, source)

    # runtime (the job script's own runtime_env) takes precedence over the
    # cluster's image-baked env
    spec.comm_env = build_comm_env([(f"{source}:image:{image_ref}", image_env), (f"{source}:job", job_env_vars)])

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec


def _fill_from_cluster(spec: JobSpec, cluster_doc: dict, source: str) -> tuple[Optional[dict], Optional[str]]:
    cluster_source = f"{source}:cluster"
    gpus_per_node = _worker_gpu_count(cluster_doc)
    max_workers = cluster_doc.get("max_workers")
    nodes = 1 + max_workers if max_workers is not None else None

    spec.gpus_per_node = resolved_or_absent(gpus_per_node, cluster_source)
    spec.nodes = resolved_or_absent(nodes, cluster_source)
    world_size = nodes * gpus_per_node if nodes is not None and gpus_per_node is not None else None
    spec.world_size = resolved_or_absent(world_size, cluster_source)

    image_ref = (cluster_doc.get("docker") or {}).get("image")
    if not image_ref:
        return None, None

    image_fields = extract_image(image_ref)
    spec.image_pin_status = resolved_or_absent(image_fields["pin_status"], f"{source}:image")
    spec.cuda_version = image_fields["cuda"]
    spec.nccl_version = image_fields["nccl"]
    spec.framework_version = image_fields["framework"]
    return image_fields["env"], image_ref


def _worker_gpu_count(cluster_doc: dict) -> Optional[int]:
    node_types = cluster_doc.get("available_node_types") or {}
    head_type = cluster_doc.get("head_node_type")

    for name, node_type in node_types.items():
        if name == head_type:
            continue
        gpu = safe_int(((node_type or {}).get("resources") or {}).get("GPU"))
        if gpu is not None:
            return gpu

    # Single-node-type cluster (or no explicit head): fall back to whatever
    # node type is there.
    for node_type in node_types.values():
        gpu = safe_int(((node_type or {}).get("resources") or {}).get("GPU"))
        if gpu is not None:
            return gpu
    return None


def _fill_from_job_py(spec: JobSpec, text: str, source: str) -> Optional[dict]:
    job_source = f"{source}:job"
    pip_field, env_vars_field, num_gpus_field = _parse_job_py(text, job_source)

    if pip_field.status == "resolved":
        spec.dependency_constraints = resolved_or_absent(parse_pip_list(pip_field.value) or None, job_source)
    else:
        spec.dependency_constraints = pip_field

    if env_vars_field.status == "resolved":
        env_vars = env_vars_field.value or {}
        spec.nccl_algo = resolved_or_absent(env_vars.get("NCCL_ALGO"), job_source)
        spec.nccl_ib_disable = resolved_or_absent(safe_int(env_vars.get("NCCL_IB_DISABLE")), job_source)
        spec.nccl_net_gdr_level = resolved_or_absent(parse_gdr_level(env_vars.get("NCCL_NET_GDR_LEVEL")), job_source)
    else:
        env_vars = None
        spec.nccl_algo = env_vars_field
        spec.nccl_ib_disable = env_vars_field
        spec.nccl_net_gdr_level = env_vars_field

    spec.launcher_nproc_per_node = num_gpus_field
    return env_vars


def _parse_job_py(text: str, source: str) -> tuple:
    absent = Field(value=None, status="absent")
    pip_field, env_vars_field, num_gpus_field = absent, absent, absent

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return pip_field, env_vars_field, num_gpus_field

    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _literal(node.value)
            if value is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = value

    runtime_env_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            runtime_env_node = _keyword_value_node(node, "runtime_env") or runtime_env_node

    if runtime_env_node is not None:
        runtime_env = _resolve(runtime_env_node, assignments)
        if isinstance(runtime_env, dict):
            pip_value = runtime_env.get("pip")
            pip_field = (
                Field(value=pip_value, status="resolved", source=source, confidence=1.0)
                if pip_value is not None
                else absent
            )
            env_vars_value = runtime_env.get("env_vars")
            env_vars_field = (
                Field(value=env_vars_value, status="resolved", source=source, confidence=1.0)
                if env_vars_value is not None
                else absent
            )
        else:
            dynamic = Field(value=None, status="unknown", reason=_DYNAMIC_RUNTIME_ENV_REASON)
            pip_field, env_vars_field = dynamic, dynamic

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not _is_ray_remote_call(decorator):
                continue
            num_gpus_node = _keyword_value_node(decorator, "num_gpus")
            if num_gpus_node is None:
                continue
            value = _resolve(num_gpus_node, assignments)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                num_gpus_field = Field(value=value, status="resolved", source=source, confidence=1.0)
            else:
                num_gpus_field = Field(value=None, status="unknown", reason=_DYNAMIC_NUM_GPUS_REASON)

    return pip_field, env_vars_field, num_gpus_field


def _is_ray_remote_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name == "remote"


def _keyword_value_node(call: ast.Call, name: str) -> Optional[ast.AST]:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _resolve(node: ast.AST, assignments: dict) -> Any:
    if isinstance(node, ast.Name) and node.id in assignments:
        return assignments[node.id]
    return _literal(node)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _find_job_py(base: Path) -> Optional[Path]:
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("*.py"))
    return candidates[0] if candidates else None


def _find_cluster_yaml(base: Path) -> Optional[Path]:
    if not base.is_dir():
        return None
    for candidate in sorted(base.glob("*.yaml")) + sorted(base.glob("*.yml")):
        doc = load_yaml_file(candidate)
        if "cluster_name" in doc or "available_node_types" in doc:
            return candidate
    return None
