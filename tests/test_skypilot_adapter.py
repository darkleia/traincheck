"""Tests for the SkyPilot task adapter."""

from pathlib import Path

from traincheck.adapters.skypilot import adapt_skypilot

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "skypilot"


def _adapt():
    return adapt_skypilot(str(EXAMPLES_DIR / "task.yaml"), base_dir=str(EXAMPLES_DIR))


def test_accelerators_split_into_gpu_type_and_gpus_per_node():
    spec = _adapt()

    assert spec.gpu_type.status == "resolved"
    assert spec.gpu_type.value == "H100"
    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8


def test_num_nodes_resolves_and_world_size_is_computed():
    spec = _adapt()

    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_image_id_strips_docker_prefix_and_resolves():
    spec = _adapt()

    assert spec.image_pin_status.status == "resolved"
    assert spec.image_pin_status.value == "pinned_soft"


def test_launcher_nproc_per_node_from_run_block():
    spec = _adapt()

    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 8


def test_reads_dependency_constraints_from_requirements_txt():
    spec = _adapt()

    assert spec.dependency_constraints.status == "resolved"
    assert spec.dependency_constraints.value == {"torch": "==2.3.0", "deepspeed": "==0.14.0"}


def test_host_env_fields_are_unknown_and_in_meta_unresolved():
    spec = _adapt()

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason

    assert len(spec.meta.unresolved) == 4
