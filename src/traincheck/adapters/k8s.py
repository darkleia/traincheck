"""Map a Kubernetes training-job CRD onto a JobSpec.

Covers the shapes of the Kubeflow Training Operator (PyTorchJob/MPIJob/
TFJob, each keyed by a *ReplicaSpecs map), Volcano's batch Job (keyed by a
`spec.tasks` list instead), and a plain batch/v1 Job. Reads placement
(nodeSelector/schedulerName/affinity/tolerations), env vars, and the
container image (via extract_image). Model config isn't inline - it's
mounted from a ConfigMap volume, so we go find that manifest in the same
repo; if it isn't there, that's reported unknown rather than guessed at.

The launcher and world_size are the more involved pieces:

- The container's `command`/`args` array is parsed as a launch line with
  the same flag parser the shell-based adapters use (`parse_launcher_tokens`),
  so a torchrun or `python -m torch.distributed.launch` invocation there is
  no longer invisible. It's already a clean argv array - no shell quoting
  to resolve - but it may reference Kubernetes' own `$(VAR_NAME)`
  substitution syntax (e.g. a per-replica `--node_rank=$(RANK)`), which is
  resolved against the container's own `env:` list the same way the shell
  extractor resolves `$VAR` against a script's own exports - and reports
  absent, not a literal "$(...)" string, for anything (like MASTER_ADDR)
  that's actually injected later by the training operator.
- PyTorchJob's nprocPerNode can be declared three different ways: the
  modern top-level `spec.nprocPerNode` (a string, also accepting
  torchrun's own host-dependent auto/cpu/gpu tokens), the deprecated
  `spec.elasticPolicy.nProcPerNode` (a plain int), or read back out of the
  command above - in that preference order. world_size is then
  sum(replicas) * that value, not replicas * the GPU resource limit - the
  GPU limit is only a resource request, not a promise about how many
  processes actually get launched per node, and is used as a last-resort
  stand-in (flagged as such) only when none of the three are set.

Kubeflow Trainer v2's `TrainJob` is a different shape entirely - handled
by its own `_adapt_trainjob` rather than forced through the v1 pod-spec
navigation above. Its container isn't inline: `spec.trainer` covers the
job-specific overrides (image, numNodes, resourcesPerNode, command/args,
env), but the base pod template - and anything `spec.trainer` doesn't
override - lives in a separately-defined runtime CR (`spec.runtimeRef`,
typically a ClusterTrainingRuntime/TrainingRuntime). If that referenced
manifest isn't one of the files traincheck was given, whatever would have
come from it (the image, if `spec.trainer.image` doesn't set one; the
launcher, if `spec.trainer.command`/`args` don't) is reported unknown with
reason "runtime CR not found" rather than absent, since it may well be set
there - we just can't check.
"""

import re
from pathlib import Path
from typing import Any, Optional

import yaml

from traincheck.extractors.image import extract_image
from traincheck.extractors.lockfile import extract_lockfile
from traincheck.extractors.shell import parse_launcher_tokens, parse_nproc_value
from traincheck.ir import Field, build_comm_env, build_launcher_fields, resolved_or_absent
from traincheck.utils import load_yaml_file, parse_gdr_level, safe_int
from traincheck.validator import JobSpec

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")

_REPLICA_SPEC_KEYS = {
    "PyTorchJob": "pytorchReplicaSpecs",
    "MPIJob": "mpiReplicaSpecs",
    "TFJob": "tfReplicaSpecs",
}

_HOST_DEPENDENT_REASON = "per-node count is host-dependent"
_RUNTIME_NOT_FOUND_REASON = "runtime CR not found"

# Kubernetes' own container command/args substitution syntax - distinct
# from a shell's $VAR/${VAR}, and resolved against the container's own
# `env:` list rather than anything exported at runtime.
_K8S_VAR_REF_RE = re.compile(r"^\$\(([A-Za-z_][A-Za-z0-9_]*)\)$")


def adapt_k8s(path: str, base_dir: str) -> JobSpec:
    doc = load_yaml_file(Path(path))

    if doc.get("kind") == "TrainJob":
        return _adapt_trainjob(doc, Path(base_dir))

    pod_spec, total_replicas = _pod_spec_and_total_replicas(doc)
    container = (pod_spec.get("containers") or [{}])[0]
    job_spec = doc.get("spec") or {}
    source = "k8s"

    spec = JobSpec()
    env_vars = _container_env(container)

    tokens = list(container.get("command") or []) + list(container.get("args") or [])
    launcher, _framework_config, _config_path, _config_overrides = parse_launcher_tokens(
        tokens, lambda value: _resolve_k8s_var(value, env_vars)
    )
    for name, launcher_field in build_launcher_fields(launcher, f"{source}:command").items():
        setattr(spec, name, launcher_field)

    # Resources
    gpu_limit, gpu_request = _gpu_resources(container)
    spec.gpus_per_node = _gpu_field(gpu_limit, gpu_request, source)
    spec.nodes = resolved_or_absent(total_replicas, source)
    spec.launcher_nproc_per_node, spec.world_size = _resolve_nproc_and_world_size(
        job_spec, launcher, total_replicas, gpu_limit, source
    )

    # Placement
    node_selector = pod_spec.get("nodeSelector") or {}
    spec.node_selector = resolved_or_absent(node_selector or None, source)
    spec.gpu_type = resolved_or_absent(node_selector.get("gpu-type"), source)
    spec.scheduler_name = resolved_or_absent(pod_spec.get("schedulerName"), source)
    spec.affinity = resolved_or_absent(pod_spec.get("affinity"), source)
    spec.tolerations = resolved_or_absent(pod_spec.get("tolerations"), source)

    # Gang scheduling
    spec.task_replicas_total = resolved_or_absent(total_replicas, source)
    spec.queue_name = resolved_or_absent(_queue_name_label(doc), source)
    spec.min_available = _resolve_min_available(doc, Path(base_dir))

    # Software: env vars straight off the container
    spec.nccl_algo = resolved_or_absent(env_vars.get("NCCL_ALGO"), source)
    spec.nccl_ib_disable = resolved_or_absent(safe_int(env_vars.get("NCCL_IB_DISABLE")), source)
    spec.nccl_net_gdr_level = resolved_or_absent(parse_gdr_level(env_vars.get("NCCL_NET_GDR_LEVEL")), source)

    # Image
    image_ref = container.get("image")
    image_env = None
    if image_ref:
        image_fields = extract_image(image_ref)
        image_env = image_fields["env"]
        spec.image_pin_status = resolved_or_absent(image_fields["pin_status"], f"{source}:image")
        spec.cuda_version = image_fields["cuda"]
        spec.nccl_version = image_fields["nccl"]
        spec.framework_version = image_fields["framework"]

    # runtime (container.env) takes precedence over image-baked env
    spec.comm_env = build_comm_env([(f"{source}:image:{image_ref}", image_env), (source, env_vars)])

    spec.dependency_constraints = resolved_or_absent(extract_lockfile(base_dir) or None, f"{source}:lockfile")

    _fill_model_config(spec, pod_spec, Path(base_dir))

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec


def _pod_spec_and_total_replicas(doc: dict) -> tuple:
    kind = doc.get("kind")
    spec = doc.get("spec") or {}

    replica_spec_key = _REPLICA_SPEC_KEYS.get(kind)
    if replica_spec_key:
        return _from_replica_specs(spec.get(replica_spec_key) or {})

    if kind == "Job":
        api_version = str(doc.get("apiVersion", ""))
        if "volcano" in api_version:
            return _from_volcano_tasks(spec.get("tasks") or [])
        pod_spec = (spec.get("template") or {}).get("spec") or {}
        return pod_spec, spec.get("parallelism") or 1

    raise ValueError(f"traincheck doesn't know how to read a k8s '{kind}' job")


def _from_replica_specs(replica_specs: dict) -> tuple:
    total_replicas = sum((role_spec or {}).get("replicas", 1) or 1 for role_spec in replica_specs.values())
    primary = replica_specs.get("Worker") or next(iter(replica_specs.values()), {})
    pod_spec = ((primary or {}).get("template") or {}).get("spec") or {}
    return pod_spec, total_replicas


def _from_volcano_tasks(tasks: list) -> tuple:
    total_replicas = sum((task or {}).get("replicas", 1) or 1 for task in tasks)
    primary = next((t for t in tasks if t.get("name") == "worker"), tasks[0] if tasks else {})
    pod_spec = ((primary or {}).get("template") or {}).get("spec") or {}
    return pod_spec, total_replicas


_KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"


def _queue_name_label(doc: dict) -> Optional[str]:
    labels = (doc.get("metadata") or {}).get("labels") or {}
    return labels.get(_KUEUE_QUEUE_LABEL)


def _resolve_min_available(doc: dict, base_dir: Path) -> Field:
    """A Volcano Job's own spec.minAvailable is inline; a PyTorchJob/
    MPIJob/TFJob's gang setting instead lives on a separately-defined
    PodGroup CR (matched here by name, the common convention when Volcano
    gang-schedules a Kubeflow-operator job) - if no such manifest is
    provided alongside the job, this comes back absent, which is exactly
    the signal the GANG-002 rule looks for (not "unknown": the rule needs
    to actually fire on a real gap, not be deferred to needs_verification).
    """
    kind = doc.get("kind")
    source = "k8s"

    if kind == "Job" and "volcano" in str(doc.get("apiVersion", "")):
        job_spec = doc.get("spec") or {}
        return resolved_or_absent(safe_int(job_spec.get("minAvailable")), source)

    if kind in _REPLICA_SPEC_KEYS:
        job_name = (doc.get("metadata") or {}).get("name")
        podgroup = _find_manifest(base_dir, "PodGroup", job_name)
        if podgroup is None:
            return Field(value=None, status="absent", source=source)
        min_available = safe_int((podgroup.get("spec") or {}).get("minAvailable"))
        return resolved_or_absent(min_available, f"{source}:podgroup:{job_name}")

    return Field(value=None, status="absent", source=source)


def _adapt_trainjob(doc: dict, base_dir: Path) -> JobSpec:
    job_spec = doc.get("spec") or {}
    trainer = job_spec.get("trainer") or {}
    source = "k8s:trainjob"

    spec = JobSpec()

    runtime_ref = job_spec.get("runtimeRef") or {}
    if runtime_ref:
        runtime_kind = runtime_ref.get("kind") or "ClusterTrainingRuntime"
        runtime_available = _find_manifest(base_dir, runtime_kind, runtime_ref.get("name")) is not None
    else:
        # No runtimeRef at all isn't a real gap to flag - there's nothing
        # to look up in the first place.
        runtime_available = True

    num_nodes = safe_int(trainer.get("numNodes"))
    spec.nodes = resolved_or_absent(num_nodes, source)

    resources_per_node = trainer.get("resourcesPerNode") or {}
    gpu_limit = safe_int((resources_per_node.get("limits") or {}).get("nvidia.com/gpu"))
    gpu_request = safe_int((resources_per_node.get("requests") or {}).get("nvidia.com/gpu"))
    if gpu_limit is None and gpu_request is None:
        # also accept a bare quantity with no requests/limits split
        gpu_limit = safe_int(resources_per_node.get("nvidia.com/gpu"))
    spec.gpus_per_node = _gpu_field(gpu_limit, gpu_request, source)

    gpus_per_node = gpu_limit if gpu_limit is not None else gpu_request
    if num_nodes is not None and gpus_per_node is not None:
        spec.world_size = resolved_or_absent(num_nodes * gpus_per_node, source)

    env_vars = {entry["name"]: entry.get("value") for entry in (trainer.get("env") or []) if entry.get("name")}

    command = list(trainer.get("command") or []) + list(trainer.get("args") or [])
    if command:
        launcher, _fc, _cp, _co = parse_launcher_tokens(command, lambda value: _resolve_k8s_var(value, env_vars))
        for name, launcher_field in build_launcher_fields(launcher, f"{source}:command").items():
            if name == "world_size":
                continue  # numNodes * resourcesPerNode's GPU count is more authoritative here
            setattr(spec, name, launcher_field)
    elif not runtime_available:
        for name in ("launcher_kind", "launcher_nnodes", "launcher_nproc_per_node"):
            setattr(spec, name, Field(value=None, status="unknown", reason=_RUNTIME_NOT_FOUND_REASON))

    image_ref = trainer.get("image")
    image_env = None
    if image_ref:
        image_fields = extract_image(image_ref)
        image_env = image_fields["env"]
        spec.image_pin_status = resolved_or_absent(image_fields["pin_status"], f"{source}:image")
        spec.cuda_version = image_fields["cuda"]
        spec.nccl_version = image_fields["nccl"]
        spec.framework_version = image_fields["framework"]
    elif not runtime_available:
        for name in ("image_pin_status", "cuda_version", "nccl_version", "framework_version"):
            setattr(spec, name, Field(value=None, status="unknown", reason=_RUNTIME_NOT_FOUND_REASON))

    spec.nccl_algo = resolved_or_absent(env_vars.get("NCCL_ALGO"), source)
    spec.nccl_ib_disable = resolved_or_absent(safe_int(env_vars.get("NCCL_IB_DISABLE")), source)
    spec.nccl_net_gdr_level = resolved_or_absent(parse_gdr_level(env_vars.get("NCCL_NET_GDR_LEVEL")), source)
    spec.comm_env = build_comm_env([(f"{source}:image:{image_ref}", image_env), (source, env_vars)])

    spec.dependency_constraints = resolved_or_absent(extract_lockfile(str(base_dir)) or None, f"{source}:lockfile")

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec


def _gpu_resources(container: dict) -> tuple[Optional[int], Optional[int]]:
    resources = container.get("resources") or {}
    limit = safe_int((resources.get("limits") or {}).get("nvidia.com/gpu"))
    request = safe_int((resources.get("requests") or {}).get("nvidia.com/gpu"))
    return limit, request


def _gpu_field(limit: Optional[int], request: Optional[int], source: str) -> Field:
    reason = ""
    if limit is not None and request is not None and limit != request:
        reason = f"resources.requests.nvidia.com/gpu ({request}) differs from resources.limits.nvidia.com/gpu ({limit})"
    value = limit if limit is not None else request
    return Field(value=value, status="resolved" if value is not None else "absent", source=source, reason=reason)


def _resolve_nproc_and_world_size(
    job_spec: dict,
    launcher: Optional[dict],
    total_replicas: int,
    gpu_limit: Optional[int],
    source: str,
) -> tuple[Field, Field]:
    """nprocPerNode can come from three places, in preference order: the
    modern top-level spec.nprocPerNode, the deprecated
    elasticPolicy.nProcPerNode, or the container's own launch command.
    world_size is replicas * that value, not replicas * the GPU resource
    limit - the limit is just a resource request, not a promise about how
    many processes actually launch per node, so it's used only as a
    stand-in (flagged as such) when nothing else resolves it, and any
    disagreement between it and a resolved nprocPerNode is flagged too.
    """
    nproc_value: Optional[int] = None
    host_dependent = False
    nproc_source = source

    raw_spec_value = job_spec.get("nprocPerNode")
    if raw_spec_value is not None:
        nproc_value, host_dependent = parse_nproc_value(str(raw_spec_value))
        nproc_source = f"{source}:spec.nprocPerNode"
    else:
        elastic_value = safe_int((job_spec.get("elasticPolicy") or {}).get("nProcPerNode"))
        if elastic_value is not None:
            nproc_value = elastic_value
            nproc_source = f"{source}:elasticPolicy.nProcPerNode"
        elif launcher is not None and launcher.get("nproc_per_node") is not None:
            nproc_value = launcher["nproc_per_node"]
            nproc_source = f"{source}:command"
        elif launcher is not None and launcher.get("nproc_per_node_host_dependent"):
            host_dependent = True
            nproc_source = f"{source}:command"

    disagreement_reason = ""
    if nproc_value is not None and gpu_limit is not None and nproc_value != gpu_limit:
        disagreement_reason = (
            f"nprocPerNode ({nproc_source}={nproc_value}) disagrees with the "
            f"nvidia.com/gpu resource limit ({gpu_limit})"
        )

    if host_dependent:
        nproc_field = Field(value=None, status="unknown", reason=_HOST_DEPENDENT_REASON)
    elif nproc_value is not None:
        nproc_field = Field(
            value=nproc_value, status="resolved", source=nproc_source, confidence=1.0, reason=disagreement_reason
        )
    elif gpu_limit is not None:
        nproc_field = Field(
            value=gpu_limit,
            status="resolved",
            source=f"{source}:gpu-limit",
            confidence=0.5,
            reason=(
                "nprocPerNode isn't set in spec.nprocPerNode, elasticPolicy, or the "
                "launch command - using the nvidia.com/gpu resource limit as a stand-in"
            ),
        )
    else:
        nproc_field = Field(value=None, status="absent", source=source)

    if nproc_field.status == "resolved":
        world_size_field = Field(
            value=total_replicas * nproc_field.value,
            status="resolved",
            source=source,
            confidence=1.0,
            reason=nproc_field.reason,
        )
    else:
        world_size_field = Field(
            value=None, status=nproc_field.status, source=nproc_field.source, reason=nproc_field.reason
        )

    return nproc_field, world_size_field


def _container_env(container: dict) -> dict:
    env = {}
    for entry in container.get("env") or []:
        name = entry.get("name")
        if name is not None:
            env[name] = entry.get("value")
    return env


def _resolve_k8s_var(value: Optional[str], env_vars: dict[str, str]) -> Optional[str]:
    """Resolve a Kubernetes `$(VAR_NAME)` command/args substitution against
    the container's own `env:` list - the same mechanism the kubelet
    itself uses, so anything not defined there (like MASTER_ADDR, which a
    training operator injects later rather than declaring statically)
    correctly comes back absent rather than a literal "$(...)" string.
    """
    if value is None:
        return None
    match = _K8S_VAR_REF_RE.match(value)
    if match:
        return env_vars.get(match.group(1))
    return None if "$(" in value else value


def _fill_model_config(spec: JobSpec, pod_spec: dict, base_dir: Path) -> None:
    configmap_name = _model_configmap_volume_name(pod_spec)
    if configmap_name is None:
        spec.model_size_billion_params = Field(
            value=None,
            status="unknown",
            reason="no configMap volume on the pod to read a model config from",
        )
        return

    configmap_doc = _find_manifest(base_dir, "ConfigMap", configmap_name)
    if configmap_doc is None:
        spec.model_size_billion_params = Field(
            value=None,
            status="unknown",
            reason=(
                f"configMap '{configmap_name}' is mounted as a volume but no matching "
                f"ConfigMap manifest was found under {base_dir}"
            ),
        )
        return

    model_doc = _embedded_yaml(configmap_doc)
    size = (model_doc.get("model") or {}).get("size_billion_params")
    spec.model_size_billion_params = resolved_or_absent(size, f"k8s:configmap:{configmap_name}")


def _model_configmap_volume_name(pod_spec: dict) -> Optional[str]:
    for volume in pod_spec.get("volumes") or []:
        config_map = volume.get("configMap") or {}
        if config_map.get("name"):
            return config_map["name"]
    return None


def _find_manifest(base_dir: Path, kind: str, name: Optional[str]) -> Optional[dict]:
    """Find a sibling manifest of the given kind/name among base_dir's own
    YAML files - used both for a PyTorchJob's ConfigMap-mounted model
    config and a TrainJob's referenced runtime CR, neither of which are
    inline in the job manifest itself.
    """
    if not name or not base_dir.is_dir():
        return None
    for candidate in sorted(base_dir.glob("*.yaml")) + sorted(base_dir.glob("*.yml")):
        doc = load_yaml_file(candidate)
        if doc.get("kind") == kind and (doc.get("metadata") or {}).get("name") == name:
            return doc
    return None


def _embedded_yaml(configmap_doc: dict) -> dict:
    for key, value in (configmap_doc.get("data") or {}).items():
        if key.endswith((".yaml", ".yml")) and isinstance(value, str):
            parsed = _safe_yaml(value)
            if isinstance(parsed, dict):
                return parsed
    return {}


def _safe_yaml(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None
