"""Tests for the Hydra config-composition extractor."""

from pathlib import Path

from traincheck.extractors.hydra import extract_hydra

CONFIG_PATH = Path(__file__).resolve().parent.parent / "examples" / "hydra" / "config.yaml"


def test_tp_resolves_from_the_stacked_defaults_file():
    result = extract_hydra(str(CONFIG_PATH))

    assert result["tensor_parallel"] == 2
    assert result["pipeline_parallel"] == 4
    assert result["data_parallel"] == 8
    assert result["sharding"] == 3
    assert result["model"] == {"name": "llama", "size_billion_params": 70}


def test_override_flips_the_stacked_value():
    result = extract_hydra(str(CONFIG_PATH), overrides=["parallelism.tensor_parallel=4"])

    assert result["tensor_parallel"] == 4
    # unrelated stacked values are untouched by the override
    assert result["pipeline_parallel"] == 4
    assert result["data_parallel"] == 8


def test_override_type_coercion_and_unrelated_dotted_key():
    result = extract_hydra(
        str(CONFIG_PATH),
        overrides=["parallelism.data_parallel=16", "model.name=mixtral"],
    )

    assert result["data_parallel"] == 16
    assert result["model"]["name"] == "mixtral"
    assert result["model"]["size_billion_params"] == 70
