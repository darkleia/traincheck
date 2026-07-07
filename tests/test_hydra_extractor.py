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


def test_group_override_loads_the_referenced_group_file():
    """`model=mixtral_8x7b` must load configs/model/mixtral_8x7b.yaml -
    not literally set `model` to the string "mixtral_8x7b".
    """
    result = extract_hydra(str(CONFIG_PATH), overrides=["model=mixtral_8x7b"])

    assert result["model"] == {"name": "mixtral", "size_billion_params": 47}
    # an unrelated group (parallelism) is untouched by the model override
    assert result["tensor_parallel"] == 2
    assert result["pipeline_parallel"] == 4


def test_value_override_still_wins_last_over_a_group_override():
    result = extract_hydra(
        str(CONFIG_PATH),
        overrides=["model=mixtral_8x7b", "model.size_billion_params=56"],
    )

    assert result["model"]["name"] == "mixtral"
    assert result["model"]["size_billion_params"] == 56


def test_dotted_key_is_never_treated_as_a_group_override():
    """`parallelism.tensor_parallel=4` must set a value even though
    "parallelism" is itself a known group name.
    """
    result = extract_hydra(str(CONFIG_PATH), overrides=["parallelism.tensor_parallel=4"])

    assert result["tensor_parallel"] == 4
    assert isinstance(result["pipeline_parallel"], int)


def test_interpolation_resolves_to_the_referenced_value(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  - _self_\n"
        "parallelism:\n  tensor_parallel: 4\n  pipeline_parallel: ${parallelism.tensor_parallel}\n"
    )

    result = extract_hydra(str(tmp_path / "config.yaml"))

    assert result["pipeline_parallel"] == 4
    assert result["tensor_parallel"] == 4


def test_interpolation_inside_a_larger_string_is_substituted(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "defaults:\n  - _self_\n"
        "model:\n  name: llama\n  size_billion_params: 70\n  tag: '${model.name}-${model.size_billion_params}b'\n"
    )

    result = extract_hydra(str(tmp_path / "config.yaml"))

    assert result["model"]["tag"] == "llama-70b"


def test_unresolvable_interpolation_is_left_as_is(tmp_path):
    (tmp_path / "config.yaml").write_text("defaults:\n  - _self_\nmodel:\n  name: ${does.not.exist}\n")

    result = extract_hydra(str(tmp_path / "config.yaml"))

    assert result["model"]["name"] == "${does.not.exist}"
