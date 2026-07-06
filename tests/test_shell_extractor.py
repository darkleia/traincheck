"""Tests for the shell/sbatch signal extractor."""

from pathlib import Path

from traincheck.extractors.shell import extract_shell

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "slurm"


def test_extract_shell_reads_slurm_sbatch_fixture():
    script_text = (EXAMPLES_DIR / "train.sbatch").read_text()

    result = extract_shell(script_text, base_dir=str(EXAMPLES_DIR))

    assert any("cuda" in m for m in result["module_loads"])
    assert any("nccl" in m for m in result["module_loads"])

    assert result["env_vars"]["NCCL_ALGO"] == "Ring"
    assert result["env_vars"]["NCCL_IB_DISABLE"] == "0"

    assert result["image_ref"] == "nvcr.io/nvidia/pytorch:24.01-py3"

    assert result["launcher"]["nnodes"] == 8
    assert result["launcher"]["nproc_per_node"] == 8

    assert result["framework_config"] == "ds_config.json"
