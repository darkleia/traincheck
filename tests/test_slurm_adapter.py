"""Tests for the Slurm sbatch adapter."""

from pathlib import Path

from traincheck.adapters.slurm import adapt_slurm

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "slurm"


def test_adapt_slurm_resolves_resources_from_sbatch_header():
    spec = adapt_slurm(str(EXAMPLES_DIR / "train.sbatch"), base_dir=str(EXAMPLES_DIR))

    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 8
    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8
    assert spec.gpu_type.status == "resolved"
    assert spec.gpu_type.value == "h100"
    assert spec.walltime.status == "resolved"
    assert spec.walltime.value == "24:00:00"
    assert spec.partition.status == "resolved"
    assert spec.partition.value == "gpu"


def test_adapt_slurm_computes_world_size_from_launcher_line():
    spec = adapt_slurm(str(EXAMPLES_DIR / "train.sbatch"), base_dir=str(EXAMPLES_DIR))

    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 64


def test_adapt_slurm_reads_nccl_algo_from_shell_body():
    spec = adapt_slurm(str(EXAMPLES_DIR / "train.sbatch"), base_dir=str(EXAMPLES_DIR))

    assert spec.nccl_algo.status == "resolved"
    assert spec.nccl_algo.value == "Ring"


def test_adapt_slurm_merges_parallelism_from_referenced_deepspeed_config():
    spec = adapt_slurm(str(EXAMPLES_DIR / "train.sbatch"), base_dir=str(EXAMPLES_DIR))

    assert spec.tensor_parallel.status == "resolved"
    assert spec.tensor_parallel.value == 2
    assert spec.pipeline_parallel.status == "resolved"
    assert spec.pipeline_parallel.value == 4
    assert spec.tensor_parallel.source.startswith("deepspeed:")


def test_adapt_slurm_resolves_the_container_image():
    """Regression test: adapt_slurm used to never call extract_image at
    all, even though train.sbatch has a --container-image= reference and
    extract_shell correctly picks it up - every other adapter resolved
    images, this one silently didn't.
    """
    spec = adapt_slurm(str(EXAMPLES_DIR / "train.sbatch"), base_dir=str(EXAMPLES_DIR))

    assert spec.image_pin_status.status == "resolved"
    assert spec.image_pin_status.value == "pinned_soft"
    # module load cuda/12.2 already resolved cuda_version; the image must
    # not clobber a more direct signal that's already there
    assert spec.cuda_version.value == "12.2"
    assert spec.cuda_version.source == "shell"


def test_adapt_slurm_reports_host_env_facts_as_unknown_and_unresolved():
    spec = adapt_slurm(str(EXAMPLES_DIR / "train.sbatch"), base_dir=str(EXAMPLES_DIR))

    host_env_fields = [spec.driver_version, spec.kernel_version, spec.ofed_version, spec.peermem_loaded]
    for host_field in host_env_fields:
        assert host_field.status == "unknown"
        assert host_field.reason

    assert len(spec.meta.unresolved) == 4
    assert all(f.status == "unknown" for f in spec.meta.unresolved)
