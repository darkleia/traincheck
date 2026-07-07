"""Tests for the Kubernetes/Kubeflow CRD adapter."""

from pathlib import Path

from traincheck.adapters.k8s import adapt_k8s

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "k8s_crd"


def _adapt():
    return adapt_k8s(str(EXAMPLES_DIR / "pytorchjob.yaml"), base_dir=str(EXAMPLES_DIR))


def test_gpus_per_pod_and_world_size():
    spec = _adapt()

    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8

    # Master (1 replica) + Worker (7 replicas) = 8 pods * 8 GPUs = 64
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_image_pin_status():
    spec = _adapt()

    assert spec.image_pin_status.status == "resolved"
    assert spec.image_pin_status.value == "pinned_soft"


def test_node_selector_and_scheduler_name():
    spec = _adapt()

    assert spec.node_selector.status == "resolved"
    assert spec.node_selector.value["gpu-type"] == "h100"
    assert spec.gpu_type.status == "resolved"
    assert spec.gpu_type.value == "h100"

    assert spec.scheduler_name.status == "resolved"
    assert spec.scheduler_name.value == "volcano"


def test_configmap_model_config_is_located_and_read():
    spec = _adapt()

    assert spec.model_size_billion_params.status == "resolved"
    assert spec.model_size_billion_params.value == 70
    assert spec.model_size_billion_params.source == "k8s:configmap:model-config"


def test_missing_configmap_manifest_is_unknown_with_reason(tmp_path):
    # Same job manifest, but pointed at an empty base_dir with no matching
    # ConfigMap manifest to be found - the model config must not be guessed.
    manifest = EXAMPLES_DIR / "pytorchjob.yaml"
    spec = adapt_k8s(str(manifest), base_dir=str(tmp_path))

    assert spec.model_size_billion_params.status == "unknown"
    assert spec.model_size_billion_params.reason


def test_host_env_fields_are_unknown_and_in_meta_unresolved():
    spec = _adapt()

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason

    assert len(spec.meta.unresolved) == 4


def test_reads_dependency_constraints_from_a_requirements_txt_in_base_dir(tmp_path):
    (tmp_path / "requirements.txt").write_text("deepspeed==0.18.5\naccelerate==1.13.0\n")

    spec = adapt_k8s(str(EXAMPLES_DIR / "pytorchjob.yaml"), base_dir=str(tmp_path))

    assert spec.dependency_constraints.status == "resolved"
    assert spec.dependency_constraints.value == {"deepspeed": "==0.18.5", "accelerate": "==1.13.0"}


def test_dependency_constraints_absent_with_no_lockfile_nearby():
    spec = _adapt()

    assert spec.dependency_constraints.status == "absent"
