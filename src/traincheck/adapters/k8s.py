"""Map a Kubernetes training-job CRD onto a JobSpec.

Covers the shapes of the Kubeflow Training Operator (PyTorchJob/MPIJob/
TFJob, each keyed by a *ReplicaSpecs map), Volcano's batch Job (keyed by a
`spec.tasks` list instead), and a plain batch/v1 Job. Reads GPU count and
total replica count for world_size, placement (nodeSelector/schedulerName/
affinity/tolerations), env vars, and the container image (via
extract_image). Model config isn't inline - it's mounted from a ConfigMap
volume, so we go find that manifest in the same repo; if it isn't there,
that's reported unknown rather than guessed at.
"""

from pathlib import Path
from typing import Any, Optional

import yaml

from traincheck.extractors.image import extract_image
from traincheck.ir import Field, build_comm_env, resolved_or_absent
from traincheck.utils import load_yaml_file, parse_gdr_level, safe_int
from traincheck.validator import JobSpec

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")

_REPLICA_SPEC_KEYS = {
    "PyTorchJob": "pytorchReplicaSpecs",
    "MPIJob": "mpiReplicaSpecs",
    "TFJob": "tfReplicaSpecs",
}


def adapt_k8s(path: str, base_dir: str) -> JobSpec:
    doc = load_yaml_file(Path(path))
    pod_spec, total_replicas = _pod_spec_and_total_replicas(doc)
    container = (pod_spec.get("containers") or [{}])[0]
    source = "k8s"

    spec = JobSpec()

    # Resources
    gpus_per_pod = _gpu_limit(container)
    spec.gpus_per_node = resolved_or_absent(gpus_per_pod, source)
    spec.nodes = resolved_or_absent(total_replicas, source)
    world_size = total_replicas * gpus_per_pod if gpus_per_pod is not None else None
    spec.world_size = resolved_or_absent(world_size, source)

    # Placement
    node_selector = pod_spec.get("nodeSelector") or {}
    spec.node_selector = resolved_or_absent(node_selector or None, source)
    spec.gpu_type = resolved_or_absent(node_selector.get("gpu-type"), source)
    spec.scheduler_name = resolved_or_absent(pod_spec.get("schedulerName"), source)
    spec.affinity = resolved_or_absent(pod_spec.get("affinity"), source)
    spec.tolerations = resolved_or_absent(pod_spec.get("tolerations"), source)

    # Software: env vars straight off the container
    env_vars = _container_env(container)
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


def _gpu_limit(container: dict) -> Optional[int]:
    limits = (container.get("resources") or {}).get("limits") or {}
    return safe_int(limits.get("nvidia.com/gpu"))


def _container_env(container: dict) -> dict:
    env = {}
    for entry in container.get("env") or []:
        name = entry.get("name")
        if name is not None:
            env[name] = entry.get("value")
    return env


def _fill_model_config(spec: JobSpec, pod_spec: dict, base_dir: Path) -> None:
    configmap_name = _model_configmap_volume_name(pod_spec)
    if configmap_name is None:
        spec.model_size_billion_params = Field(
            value=None,
            status="unknown",
            reason="no configMap volume on the pod to read a model config from",
        )
        return

    configmap_doc = _find_configmap_manifest(base_dir, configmap_name)
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


def _find_configmap_manifest(base_dir: Path, name: str) -> Optional[dict]:
    if not base_dir.is_dir():
        return None
    for candidate in sorted(base_dir.glob("*.yaml")) + sorted(base_dir.glob("*.yml")):
        doc = load_yaml_file(candidate)
        if doc.get("kind") == "ConfigMap" and (doc.get("metadata") or {}).get("name") == name:
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
