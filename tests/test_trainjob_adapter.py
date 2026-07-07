"""Tests for Kubeflow Trainer v2's TrainJob adapter."""

from pathlib import Path

import yaml

from traincheck.adapters.k8s import adapt_k8s
from traincheck.detect import Stack, detect_stack

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "trainjob"


def _trainjob_doc(trainer=None, runtime_ref=None):
    doc = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {"name": "test"},
        "spec": {"trainer": trainer or {}},
    }
    if runtime_ref is not None:
        doc["spec"]["runtimeRef"] = runtime_ref
    return doc


def _write(tmp_path, doc, name="trainjob.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return str(path), str(tmp_path)


def test_detect_stack_recognizes_trainjob():
    assert detect_stack(EXAMPLES_DIR / "trainjob.yaml") == Stack.K8S_CRD


def test_numnodes_and_resources_per_node_are_read():
    spec = adapt_k8s(str(EXAMPLES_DIR / "trainjob.yaml"), base_dir=str(EXAMPLES_DIR))

    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8
    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_command_array_is_parsed_as_a_launcher():
    spec = adapt_k8s(str(EXAMPLES_DIR / "trainjob.yaml"), base_dir=str(EXAMPLES_DIR))

    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "torchrun"
    assert spec.launcher_nnodes.value == 8
    assert spec.launcher_nproc_per_node.value == 8


def test_image_is_read_from_trainer_spec():
    spec = adapt_k8s(str(EXAMPLES_DIR / "trainjob.yaml"), base_dir=str(EXAMPLES_DIR))

    assert spec.image_pin_status.status == "resolved"
    assert spec.image_pin_status.value == "pinned_soft"


def test_env_vars_are_read_from_trainer_spec():
    spec = adapt_k8s(str(EXAMPLES_DIR / "trainjob.yaml"), base_dir=str(EXAMPLES_DIR))

    assert spec.nccl_algo.status == "resolved"
    assert spec.nccl_algo.value == "Ring"


def test_absent_runtime_ref_yields_unknown_image_and_launcher_fields(tmp_path):
    """No image/command in trainer spec, and no ClusterTrainingRuntime
    manifest sitting alongside it to fall back on - those fields would
    normally come from the runtime CR, so they must be reported unknown,
    not silently absent, since the runtime might well set them.
    """
    doc = _trainjob_doc(
        trainer={"numNodes": 4, "resourcesPerNode": {"limits": {"nvidia.com/gpu": 8}}},
        runtime_ref={"name": "torch-distributed", "kind": "ClusterTrainingRuntime"},
    )

    spec = adapt_k8s(*_write(tmp_path, doc))

    for name in ("image_pin_status", "cuda_version", "nccl_version", "framework_version"):
        field = getattr(spec, name)
        assert field.status == "unknown", name
        assert field.reason == "runtime CR not found"

    for name in ("launcher_kind", "launcher_nnodes", "launcher_nproc_per_node"):
        field = getattr(spec, name)
        assert field.status == "unknown", name
        assert field.reason == "runtime CR not found"

    # numNodes/resourcesPerNode are still read normally regardless
    assert spec.nodes.value == 4
    assert spec.gpus_per_node.value == 8
    assert spec.world_size.value == 32


def test_runtime_ref_present_does_not_flag_the_missing_runtime_reason(tmp_path):
    """When the referenced runtime CR *is* found alongside the TrainJob,
    the gap isn't "the runtime is missing" anymore - traincheck doesn't
    parse the runtime's own pod template, so the fields just come back
    absent (nothing else set them), not unknown-because-not-found.
    """
    runtime_doc = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "ClusterTrainingRuntime",
        "metadata": {"name": "torch-distributed"},
        "spec": {},
    }
    (tmp_path / "runtime.yaml").write_text(yaml.safe_dump(runtime_doc))

    doc = _trainjob_doc(
        trainer={"numNodes": 4, "resourcesPerNode": {"limits": {"nvidia.com/gpu": 8}}},
        runtime_ref={"name": "torch-distributed", "kind": "ClusterTrainingRuntime"},
    )
    path, base_dir = _write(tmp_path, doc)

    spec = adapt_k8s(path, base_dir)

    assert spec.image_pin_status.status == "absent"
    assert spec.launcher_kind.status == "absent"


def test_no_runtime_ref_at_all_is_not_treated_as_a_missing_runtime(tmp_path):
    doc = _trainjob_doc(trainer={"numNodes": 4, "resourcesPerNode": {"limits": {"nvidia.com/gpu": 8}}})

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.image_pin_status.status == "absent"
    assert spec.launcher_kind.status == "absent"


def test_host_env_fields_are_unknown_and_in_meta_unresolved():
    spec = adapt_k8s(str(EXAMPLES_DIR / "trainjob.yaml"), base_dir=str(EXAMPLES_DIR))

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason

    assert len(spec.meta.unresolved) == 4


def test_reads_dependency_constraints_from_a_requirements_txt_in_base_dir(tmp_path):
    doc = _trainjob_doc(trainer={"numNodes": 2, "resourcesPerNode": {"limits": {"nvidia.com/gpu": 8}}})
    path, base_dir = _write(tmp_path, doc)
    (tmp_path / "requirements.txt").write_text("transformers==5.10.1\n")

    spec = adapt_k8s(path, base_dir)

    assert spec.dependency_constraints.status == "resolved"
    assert spec.dependency_constraints.value == {"transformers": "==5.10.1"}
