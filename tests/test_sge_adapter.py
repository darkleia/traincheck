"""Tests for the SGE (qsub) adapter."""

from pathlib import Path

from traincheck.adapters.sge import adapt_sge

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "sge"


def _adapt():
    return adapt_sge(str(EXAMPLES_DIR / "train.sge"), base_dir=str(EXAMPLES_DIR))


def test_adapt_sge_derives_nodes_from_pe_slots_and_gpu_count():
    """-pe mpi 64 (total slots) and -l gpu=8 (per-node GPU count) together
    imply 8 nodes; world_size uses the slot total directly.
    """
    spec = _adapt()

    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8
    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_adapt_sge_resolves_walltime_and_queue():
    spec = _adapt()

    assert spec.walltime.status == "resolved"
    assert spec.walltime.value == "24:00:00"
    assert spec.partition.status == "resolved"
    assert spec.partition.value == "gpu.q"


def test_adapt_sge_extracts_the_bodys_launcher():
    spec = _adapt()

    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "torchrun"
    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 8
    assert spec.nccl_algo.status == "resolved"
    assert spec.nccl_algo.value == "Ring"


def test_adapt_sge_reports_host_env_facts_as_unknown():
    spec = _adapt()

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason
