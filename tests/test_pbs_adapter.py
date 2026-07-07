"""Tests for the PBS (Torque/PBS Pro) adapter."""

from pathlib import Path

from traincheck.adapters.pbs import adapt_pbs

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "pbs"


def _adapt():
    return adapt_pbs(str(EXAMPLES_DIR / "train.pbs"), base_dir=str(EXAMPLES_DIR))


def test_adapt_pbs_resolves_resources_from_the_header():
    """-l select=8:ngpus=8:mpiprocs=8 packs node count, GPU count, and an
    unrelated key into one colon-chained value - only select/ngpus matter.
    """
    spec = _adapt()

    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8
    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_adapt_pbs_walltime_survives_its_own_colon_format():
    """walltime's value ("24:00:00") contains colons that are part of the
    HH:MM:SS format, not `-l`'s key=value chunk separator - regression
    coverage for that ambiguity.
    """
    spec = _adapt()

    assert spec.walltime.status == "resolved"
    assert spec.walltime.value == "24:00:00"


def test_adapt_pbs_resolves_queue():
    spec = _adapt()

    assert spec.partition.status == "resolved"
    assert spec.partition.value == "gpu"


def test_adapt_pbs_extracts_the_bodys_launcher():
    spec = _adapt()

    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "torchrun"
    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 8
    assert spec.nccl_algo.status == "resolved"
    assert spec.nccl_algo.value == "Ring"


def test_adapt_pbs_reports_host_env_facts_as_unknown():
    spec = _adapt()

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason
