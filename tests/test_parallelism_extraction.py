"""Tests for Megatron-LM launch-flag parallelism and FSDP sharding
strategy extraction, the tp*pp world_size divisibility check, and
data_parallel derivation.
"""

import json

import yaml

from traincheck.adapters.bare import adapt_bare
from traincheck.core import RuleEngine
from traincheck.ir import Field
from traincheck.rules import BUILTIN_RULES
from traincheck.validator import JobSpec, Validator


def _write(tmp_path, body: str, name: str = "run.sh"):
    script = tmp_path / name
    script.write_text(body)
    return str(script), str(tmp_path)


def _fired_ids(spec):
    engine = RuleEngine()
    for rule in BUILTIN_RULES:
        engine.register(rule)
    result = engine.check(vars(spec))
    return {v.rule.id for v in result.violations}


# --- Megatron-LM flags ---------------------------------------------------


def test_megatron_tensor_and_pipeline_parallel_size_are_parsed(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "torchrun --nnodes=2 --nproc-per-node=8 pretrain_gpt.py "
        "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 4\n",
    )

    spec = adapt_bare(path, base)

    assert spec.tensor_parallel.status == "resolved"
    assert spec.tensor_parallel.value == 2
    assert spec.pipeline_parallel.status == "resolved"
    assert spec.pipeline_parallel.value == 4


def test_megatron_expert_and_context_parallel_size_are_parsed(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "torchrun --nnodes=1 --nproc-per-node=8 pretrain_gpt.py "
        "--expert-model-parallel-size 2 --context-parallel-size 2\n",
    )

    spec = adapt_bare(path, base)

    assert spec.expert_parallel.status == "resolved"
    assert spec.expert_parallel.value == 2
    assert spec.context_parallel.status == "resolved"
    assert spec.context_parallel.value == 2


def test_megatron_sequence_parallel_switch_is_parsed(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\ntorchrun --nnodes=1 --nproc-per-node=8 pretrain_gpt.py --sequence-parallel\n",
    )

    spec = adapt_bare(path, base)

    assert spec.sequence_parallel.status == "resolved"
    assert spec.sequence_parallel.value is True


def test_sequence_parallel_absent_when_flag_not_given(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\ntorchrun --nnodes=1 --nproc-per-node=8 pretrain_gpt.py\n")

    spec = adapt_bare(path, base)

    assert spec.sequence_parallel.status == "absent"


def test_megatron_underscore_flag_spellings_also_parse(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "torchrun --nnodes=1 --nproc-per-node=8 pretrain_gpt.py "
        "--tensor_model_parallel_size 2 --pipeline_model_parallel_size 2\n",
    )

    spec = adapt_bare(path, base)

    assert spec.tensor_parallel.value == 2
    assert spec.pipeline_parallel.value == 2


# --- tp*pp world_size divisibility check ---------------------------------


def test_tp_pp_not_dividing_world_size_is_flagged(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "torchrun --nnodes=1 --nproc-per-node=10 pretrain_gpt.py "
        "--tensor-model-parallel-size 3 --pipeline-model-parallel-size 1\n",
    )

    spec = adapt_bare(path, base)
    result = Validator().validate_spec(spec)

    assert spec.world_size.value == 10
    assert "PARALLEL-002" in {v.rule.id for v in result.violations}
    # a floor-divided data_parallel (10 // 3 == 3) would be misleading -
    # 3*3 != 10 - so it must stay unresolved, not silently wrong
    assert spec.data_parallel.status != "resolved"


def test_tp_pp_dividing_world_size_evenly_does_not_fire():
    spec = JobSpec()
    spec.tensor_parallel = Field(2, status="resolved", source="test")
    spec.pipeline_parallel = Field(4, status="resolved", source="test")
    spec.world_size = Field(64, status="resolved", source="test")

    assert "PARALLEL-002" not in _fired_ids(spec)


def test_data_parallel_computed_when_world_size_and_tp_pp_are_known(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "torchrun --nnodes=1 --nproc-per-node=8 pretrain_gpt.py "
        "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2\n",
    )

    spec = adapt_bare(path, base)

    assert spec.world_size.value == 8
    assert spec.data_parallel.status == "resolved"
    assert spec.data_parallel.value == 2  # 8 / (2*2)


# --- FSDP sharding strategy (Accelerate config) ---------------------------


def _write_accelerate_config(tmp_path, sharding_strategy, distributed_type="FSDP", name="accelerate_config.yaml"):
    doc = {"distributed_type": distributed_type, "fsdp_config": {"fsdp_sharding_strategy": sharding_strategy}}
    (tmp_path / name).write_text(yaml.safe_dump(doc))


def test_fsdp_sharding_strategy_string_is_mapped(tmp_path):
    _write_accelerate_config(tmp_path, "FULL_SHARD")
    path, base = _write(tmp_path, "#!/bin/bash\naccelerate launch --config_file accelerate_config.yaml train.py\n")

    spec = adapt_bare(path, base)

    assert spec.sharding.status == "resolved"
    assert spec.sharding.value == "FULL_SHARD"


def test_fsdp_sharding_strategy_legacy_integer_form_is_mapped(tmp_path):
    _write_accelerate_config(tmp_path, 2)  # 2 -> SHARD_GRAD_OP
    path, base = _write(tmp_path, "#!/bin/bash\naccelerate launch --config_file accelerate_config.yaml train.py\n")

    spec = adapt_bare(path, base)

    assert spec.sharding.value == "SHARD_GRAD_OP"


def test_fsdp_configured_only_in_python_is_unknown(tmp_path):
    """accelerate launch with no --config_file at all - FSDP might be set
    up directly in the training script, or via Accelerate's own default
    config, neither of which is readable statically.
    """
    path, base = _write(tmp_path, "#!/bin/bash\naccelerate launch train.py\n")

    spec = adapt_bare(path, base)

    assert spec.sharding.status == "unknown"
    assert "Python" in spec.sharding.reason


def test_fsdp_config_referenced_but_missing_file_is_unknown(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\naccelerate launch --config_file missing.yaml train.py\n")

    spec = adapt_bare(path, base)

    assert spec.sharding.status == "unknown"


def test_non_fsdp_accelerate_config_does_not_flag_sharding(tmp_path):
    """A plain multi-GPU Accelerate config (no FSDP at all) shouldn't be
    treated as a gap - the job simply isn't using FSDP.
    """
    _write_accelerate_config(tmp_path, None, distributed_type="MULTI_GPU")
    path, base = _write(tmp_path, "#!/bin/bash\naccelerate launch --config_file accelerate_config.yaml train.py\n")

    spec = adapt_bare(path, base)

    assert spec.sharding.status == "absent"


def test_fsdp_does_not_flag_when_launcher_is_not_accelerate(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\ntorchrun --nnodes=1 --nproc-per-node=8 train.py\n")

    spec = adapt_bare(path, base)

    assert spec.sharding.status == "absent"


# --- Megatron + DeepSpeed combined (Megatron-DeepSpeed) -------------------


def test_megatron_flags_survive_a_deepspeed_config_that_does_not_set_tp_pp(tmp_path):
    ds_config = {
        "train_micro_batch_size_per_gpu": 4,
        "zero_optimization": {"stage": 1},
    }
    (tmp_path / "ds_config.json").write_text(json.dumps(ds_config))

    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "torchrun --nnodes=1 --nproc-per-node=8 pretrain_gpt.py "
        "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 "
        "--deepspeed ds_config.json\n",
    )

    spec = adapt_bare(path, base)

    # the DeepSpeed config's own zero stage still wins for sharding...
    assert spec.sharding.value == 1
    # ...but tp/pp came from Megatron's launch flags, since the ds_config
    # doesn't set them itself - they must not have been clobbered to absent
    assert spec.tensor_parallel.value == 2
    assert spec.pipeline_parallel.value == 2
