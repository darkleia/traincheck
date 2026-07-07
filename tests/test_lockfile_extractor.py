"""Tests for the dependency-lockfile extractor."""

from pathlib import Path

from traincheck.extractors.lockfile import extract_lockfile

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "examples"


def test_finds_torch_with_exact_constraint_in_slurm_requirements():
    constraints = extract_lockfile(str(EXAMPLES_ROOT / "slurm"))

    assert constraints["torch"] == "==2.3.0"
    assert constraints["deepspeed"] == "==0.14.0"


def test_finds_torch_with_exact_constraint_in_ray_requirements():
    constraints = extract_lockfile(str(EXAMPLES_ROOT / "ray"))

    assert constraints["torch"] == "==2.3.0"


def test_toml_lock_fixture_parses():
    constraints = extract_lockfile(str(EXAMPLES_ROOT / "lockfiles"))

    assert constraints["torch"] == "2.3.0"
    assert constraints["nvidia-nccl-cu12"] == "2.19.3"
    assert constraints["transformers"] == "4.38.0"


def test_untracked_package_is_not_included():
    constraints = extract_lockfile(str(EXAMPLES_ROOT / "ray"))

    assert "ray" not in constraints


def test_accelerate_and_megatron_core_are_tracked(tmp_path):
    (tmp_path / "requirements.txt").write_text("accelerate==1.13.0\nmegatron_core==0.9.0\n")

    constraints = extract_lockfile(str(tmp_path))

    assert constraints["accelerate"] == "==1.13.0"
    # underscore-vs-hyphen naming must normalize the same way pip itself does
    assert constraints["megatron_core"] == "==0.9.0"
