"""Tests for the Ray cluster/job adapter."""

from pathlib import Path

from traincheck.adapters.ray import adapt_ray

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "ray"


def test_cluster_yaml_resolves_image_and_resources():
    spec = adapt_ray(str(EXAMPLES_DIR / "cluster.yaml"), base_dir=str(EXAMPLES_DIR))

    assert spec.image_pin_status.status == "resolved"
    assert spec.image_pin_status.value == "pinned_soft"

    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8
    # 1 head node + 7 workers
    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64

    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "ray"


def test_job_py_finds_pip_and_env_vars_from_runtime_env():
    spec = adapt_ray(str(EXAMPLES_DIR / "job.py"), base_dir=str(EXAMPLES_DIR))

    assert spec.dependency_constraints.status == "resolved"
    assert spec.dependency_constraints.value == {"torch": "==2.3.0", "transformers": "==4.38.0"}

    assert spec.nccl_algo.status == "resolved"
    assert spec.nccl_algo.value == "Ring"
    assert spec.nccl_ib_disable.status == "resolved"
    assert spec.nccl_ib_disable.value == 0


def test_job_py_finds_ray_remote_num_gpus():
    spec = adapt_ray(str(EXAMPLES_DIR / "job.py"), base_dir=str(EXAMPLES_DIR))

    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 1
    assert spec.launcher_kind.value == "ray"


def test_either_entrypoint_pulls_in_its_sibling():
    # cluster.yaml as path should also find job.py's runtime_env, and vice
    # versa, since adapt_ray looks for the counterpart in base_dir either way.
    from_cluster = adapt_ray(str(EXAMPLES_DIR / "cluster.yaml"), base_dir=str(EXAMPLES_DIR))
    from_job = adapt_ray(str(EXAMPLES_DIR / "job.py"), base_dir=str(EXAMPLES_DIR))

    assert from_cluster.dependency_constraints.status == "resolved"
    assert from_job.image_pin_status.status == "resolved"


def test_dynamic_runtime_env_is_unknown_not_absent(tmp_path):
    dynamic_job = tmp_path / "job.py"
    dynamic_job.write_text(
        "import ray\ndef build_env():\n    return {'pip': ['torch']}\nray.init(runtime_env=build_env())\n"
    )

    spec = adapt_ray(str(dynamic_job), base_dir=str(tmp_path))

    assert spec.dependency_constraints.status == "unknown"
    assert spec.dependency_constraints.reason
    assert spec.nccl_algo.status == "unknown"


def test_host_env_fields_are_unknown_and_in_meta_unresolved():
    spec = adapt_ray(str(EXAMPLES_DIR / "cluster.yaml"), base_dir=str(EXAMPLES_DIR))

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason

    assert len(spec.meta.unresolved) == 4
