"""Tests for the DeepSpeed config adapter."""

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
