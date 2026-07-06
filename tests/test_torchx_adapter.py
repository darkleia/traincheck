"""Tests for the torchx adapter."""

from pathlib import Path

from traincheck.adapters.torchx import adapt_torchx

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "torchx"


def _adapt():
    return adapt_torchx(str(EXAMPLES_DIR / "run.sh"), base_dir=str(EXAMPLES_DIR))


def test_scheduler_resolves_to_slurm_from_run_line():
    spec = _adapt()

    assert spec.meta.stack.status == "resolved"
    assert spec.meta.stack.value == "slurm"


def test_job_geometry_parsed_from_dash_j():
    spec = _adapt()

    assert spec.launcher_nnodes.status == "resolved"
    assert spec.launcher_nnodes.value == 8
    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 8
    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64
    assert spec.launcher_kind.value == "torchx"


def test_delegation_to_slurm_populates_a_software_field():
    spec = _adapt()

    # nccl_algo lives in the Framework/Software section of JobSpec, and
    # only extract_shell (run via the delegated adapt_slurm) could have
    # found it - it's not something adapt_torchx reads itself.
    assert spec.nccl_algo.status == "resolved"
    assert spec.nccl_algo.value == "Ring"
    assert spec.nccl_ib_disable.status == "resolved"
    assert spec.nccl_ib_disable.value == 0


def test_delegation_does_not_clobber_torchx_geometry():
    spec = _adapt()

    # adapt_slurm sees no #SBATCH in run.sh, so its own Resources.nodes is
    # absent - that must not overwrite torchx's own -j-derived geometry.
    assert spec.launcher_nnodes.value == 8
    assert spec.launcher_nproc_per_node.value == 8


def test_deepspeed_config_still_merges_through_the_delegate():
    spec = _adapt()

    assert spec.tensor_parallel.status == "resolved"
    assert spec.tensor_parallel.value == 2
    assert spec.pipeline_parallel.status == "resolved"
    assert spec.pipeline_parallel.value == 4


def test_scheduler_falls_back_to_torchxconfig_cli_run_section(tmp_path):
    (tmp_path / "run.sh").write_text(
        "#!/bin/bash\ntorchx run dist.ddp -j 8x8 --script train.py\n"
    )
    (tmp_path / ".torchxconfig").write_text("[cli:run]\nscheduler = slurm\n")

    spec = adapt_torchx(str(tmp_path / "run.sh"), base_dir=str(tmp_path))

    assert spec.meta.stack.status == "resolved"
    assert spec.meta.stack.value == "slurm"


def test_no_scheduler_found_marks_meta_stack_unknown(tmp_path):
    (tmp_path / "run.sh").write_text("#!/bin/bash\ntorchx run dist.ddp -j 8x8\n")

    spec = adapt_torchx(str(tmp_path / "run.sh"), base_dir=str(tmp_path))

    assert spec.meta.stack.status == "unknown"
    assert spec.meta.stack.reason
    assert spec.meta.stack in spec.meta.unresolved
