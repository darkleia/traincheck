"""Tests for the submitit AutoExecutor adapter."""

from pathlib import Path

from traincheck.adapters.submitit import adapt_submitit

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "submitit"


def _adapt():
    return adapt_submitit(str(EXAMPLES_DIR / "job.py"))


def test_nodes_and_gpus_per_node_resolve():
    spec = _adapt()

    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8
    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8


def test_walltime_from_timeout_min():
    spec = _adapt()

    assert spec.walltime.status == "resolved"
    assert spec.walltime.value == 1440


def test_partition_from_slurm_partition():
    spec = _adapt()

    assert spec.partition.status == "resolved"
    assert spec.partition.value == "gpu"


def test_meta_stack_is_submitit_with_no_local_fallback_note():
    spec = _adapt()

    assert spec.meta.stack.status == "resolved"
    assert spec.meta.stack.value == "submitit"
    assert not spec.meta.stack.reason


def test_local_backend_is_noted_when_no_slurm_kwargs(tmp_path):
    local_job = tmp_path / "job.py"
    local_job.write_text(
        "import submitit\n"
        "executor = submitit.AutoExecutor(folder='logs')\n"
        "executor.update_parameters(nodes=1, gpus_per_node=1, timeout_min=60)\n"
    )

    spec = adapt_submitit(str(local_job))

    assert spec.nodes.value == 1
    assert spec.meta.stack.value == "submitit"
    assert spec.meta.stack.reason
    assert "local" in spec.meta.stack.reason.lower()


def test_no_update_parameters_call_leaves_fields_absent(tmp_path):
    empty_job = tmp_path / "job.py"
    empty_job.write_text("import submitit\n")

    spec = adapt_submitit(str(empty_job))

    assert spec.nodes.status == "absent"
    assert spec.meta.stack.value == "submitit"
    assert spec.meta.stack.reason
