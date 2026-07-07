"""Tests for the bare-metal (no-scheduler) adapter."""

from pathlib import Path

from traincheck.adapters.bare import adapt_bare

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "bare"


def test_gdr_level_named_form_resolves_not_lost_to_int_cast(tmp_path):
    """Regression test: NCCL_NET_GDR_LEVEL accepts named distances
    (LOC/PIX/PXB/PHB/SYS) as well as numbers - these used to be silently
    dropped to "absent" by an int() cast that only handled the numeric form.
    """
    script = tmp_path / "run.sh"
    script.write_text(
        "#!/bin/bash\nexport NCCL_NET_GDR_LEVEL=PXB\ntorchrun --nnodes=1 --nproc-per-node=8 train.py\n"
    )

    spec = adapt_bare(str(script), base_dir=str(tmp_path))

    assert spec.nccl_net_gdr_level.status == "resolved"
    assert spec.nccl_net_gdr_level.value == "PXB"


def _adapt():
    return adapt_bare(str(EXAMPLES_DIR / "run.sh"), base_dir=str(EXAMPLES_DIR))


def test_launcher_nnodes_and_nproc_resolve_from_the_command():
    spec = _adapt()

    assert spec.launcher_nnodes.status == "resolved"
    assert spec.launcher_nnodes.value == 8
    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 8
    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "torchrun"
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_deepspeed_config_resolves():
    spec = _adapt()

    assert spec.tensor_parallel.status == "resolved"
    assert spec.tensor_parallel.value == 2
    assert spec.pipeline_parallel.status == "resolved"
    assert spec.pipeline_parallel.value == 4
    assert spec.sharding.status == "resolved"
    assert spec.sharding.value == 3


def test_all_resources_fields_are_unknown_with_no_scheduler_reason():
    spec = _adapt()

    for name in (
        "nodes",
        "gpus_per_node",
        "gpu_type",
        "interconnect",
        "gpu_memory_gb",
        "walltime",
        "partition",
    ):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason == "no scheduler in entrypoint"


def test_no_image_referenced_so_image_fields_stay_unset():
    spec = _adapt()

    # run.sh never references a container image at all
    assert spec.image_pin_status.status == "absent"


def test_host_env_fields_are_unknown_and_in_meta_unresolved():
    spec = _adapt()

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason

    assert len(spec.meta.unresolved) == 4
