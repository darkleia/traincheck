"""Sanity checks for the example fixtures under examples/."""

import json
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "slurm"


def test_sbatch_script_exists_and_parses_as_text():
    path = EXAMPLES_DIR / "train.sbatch"
    assert path.exists()
    text = path.read_text()
    assert text.strip()
    assert "#SBATCH" in text


def test_train_py_exists_and_is_non_empty():
    path = EXAMPLES_DIR / "train.py"
    assert path.exists()
    assert path.read_text().strip()


def test_requirements_txt_exists_and_is_non_empty():
    path = EXAMPLES_DIR / "requirements.txt"
    assert path.exists()
    assert path.read_text().strip()


def test_ds_config_exists_and_parses_as_json():
    path = EXAMPLES_DIR / "ds_config.json"
    assert path.exists()
    text = path.read_text()
    assert text.strip()

    config = json.loads(text)
    assert isinstance(config, dict)
    assert config
