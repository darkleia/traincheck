"""Tests for the LSF (bsub) adapter."""

from pathlib import Path

from traincheck.adapters.lsf import adapt_lsf

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "lsf"


def _adapt():
    return adapt_lsf(str(EXAMPLES_DIR / "train.lsf"), base_dir=str(EXAMPLES_DIR))


def test_adapt_lsf_derives_nodes_from_ntasks_and_span_ptile():
    """No -nnodes in this fixture - nodes has to come from -n 64 (total
    tasks) divided by -R "span[ptile=8]" (tasks per host).
    """
    spec = _adapt()

    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8


def test_adapt_lsf_reads_gpu_count_from_quoted_gpu_flag():
    spec = _adapt()

    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_adapt_lsf_resolves_walltime_and_queue():
    spec = _adapt()

    assert spec.walltime.status == "resolved"
    assert spec.walltime.value == "24:00"
    assert spec.partition.status == "resolved"
    assert spec.partition.value == "gpu"


def test_adapt_lsf_extracts_the_bodys_launcher():
    spec = _adapt()

    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "torchrun"
    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 8
    assert spec.nccl_algo.status == "resolved"
    assert spec.nccl_algo.value == "Ring"


def test_adapt_lsf_reports_host_env_facts_as_unknown():
    spec = _adapt()

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason
