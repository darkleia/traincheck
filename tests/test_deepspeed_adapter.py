"""Tests for the DeepSpeed config adapter."""

import json
from pathlib import Path

from traincheck.adapters.deepspeed import adapt_deepspeed

DS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "examples" / "slurm" / "ds_config.json"


def test_adapt_deepspeed_reads_parallelism_and_batch_fields():
    fields = adapt_deepspeed(str(DS_CONFIG_PATH))

    assert fields["sharding"].value == 3
    assert fields["tensor_parallel"].value == 2
    assert fields["pipeline_parallel"].value == 4
    assert fields["train_micro_batch_size_per_gpu"].value == 4
    assert fields["gradient_accumulation_steps"].value == 8

    resolved_keys = (
        "sharding",
        "tensor_parallel",
        "pipeline_parallel",
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
    )
    for key in resolved_keys:
        field = fields[key]
        assert field.status == "resolved"
        assert field.source == f"deepspeed:{DS_CONFIG_PATH}"


def test_adapt_deepspeed_missing_key_is_absent_not_unknown():
    fields = adapt_deepspeed(str(DS_CONFIG_PATH))

    data_parallel = fields["data_parallel"]
    assert data_parallel.status == "absent"
    assert data_parallel.value is None


def test_adapt_deepspeed_auto_value_is_unknown_not_a_literal_string(tmp_path):
    """Regression test: DeepSpeed/HF Trainer fill "auto" fields in at
    runtime from their own args - reading "auto" back as a resolved value
    is actively wrong, not just incomplete.
    """
    config_path = tmp_path / "ds_config.json"
    config_path.write_text(
        json.dumps(
            {
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": 8,
                "zero_optimization": {"stage": "auto"},
                "tensor_parallel_size": 2,
            }
        )
    )

    fields = adapt_deepspeed(str(config_path))

    assert fields["train_micro_batch_size_per_gpu"].status == "unknown"
    assert fields["train_micro_batch_size_per_gpu"].value is None
    assert fields["train_micro_batch_size_per_gpu"].reason

    assert fields["sharding"].status == "unknown"
    assert fields["sharding"].reason

    # real values elsewhere in the same file are unaffected
    assert fields["gradient_accumulation_steps"].status == "resolved"
    assert fields["gradient_accumulation_steps"].value == 8
    assert fields["tensor_parallel"].status == "resolved"
    assert fields["tensor_parallel"].value == 2
