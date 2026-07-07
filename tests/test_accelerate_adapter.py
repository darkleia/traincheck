"""Tests for the Accelerate adapter: a bare default_config.yaml as the
entrypoint, and an `accelerate launch` line (in a shell script) whose own
flags override the same file.
"""

from pathlib import Path

import yaml

from traincheck.adapters.accelerate import adapt_accelerate
from traincheck.adapters.bare import adapt_bare
from traincheck.detect import Stack, detect_stack

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "accelerate"


def _adapt():
    return adapt_accelerate(str(EXAMPLES_DIR / "default_config.yaml"), base_dir=str(EXAMPLES_DIR))


def test_detect_stack_recognizes_an_accelerate_config():
    assert detect_stack(EXAMPLES_DIR / "default_config.yaml") == Stack.ACCELERATE


def test_num_processes_becomes_world_size():
    spec = _adapt()

    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 16


def test_num_machines_and_derived_nproc_per_node():
    spec = _adapt()

    assert spec.launcher_nnodes.value == 2
    assert spec.launcher_nproc_per_node.value == 8  # 16 / 2


def test_machine_rank_and_main_process_address_are_read():
    spec = _adapt()

    assert spec.launcher_node_rank.value == 0
    assert spec.launcher_master_addr.value == "10.0.0.1"
    assert spec.launcher_master_port.value == 29500


def test_compute_environment_distributed_type_mixed_precision_gpu_ids():
    spec = _adapt()

    assert spec.compute_environment.value == "LOCAL_MACHINE"
    assert spec.distributed_type.value == "DEEPSPEED"
    assert spec.mixed_precision.value == "bf16"
    assert spec.gpu_ids.value == "all"


def test_embedded_deepspeed_block_maps_zero_stage_and_offload():
    spec = _adapt()

    assert spec.sharding.status == "resolved"
    assert spec.sharding.value == 2
    assert spec.zero_offload.status == "resolved"
    assert spec.zero_offload.value == {"optimizer": "cpu", "param": "cpu"}
    assert spec.gradient_accumulation_steps.status == "resolved"
    assert spec.gradient_accumulation_steps.value == 4


def test_launcher_kind_is_accelerate():
    spec = _adapt()

    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "accelerate"


def test_host_env_fields_are_unknown_and_in_meta_unresolved():
    spec = _adapt()

    for name in ("driver_version", "kernel_version", "ofed_version", "peermem_loaded"):
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason

    assert len(spec.meta.unresolved) == 4


# --- embedded FSDP block ---------------------------------------------------


def _write(tmp_path, doc, name="default_config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return str(path), str(tmp_path)


def test_embedded_fsdp_block_maps_sharding_strategy(tmp_path):
    doc = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "num_processes": 8,
        "num_machines": 1,
        "fsdp_config": {"fsdp_sharding_strategy": "FULL_SHARD"},
    }

    spec = adapt_accelerate(*_write(tmp_path, doc))

    assert spec.sharding.status == "resolved"
    assert spec.sharding.value == "FULL_SHARD"
    assert spec.world_size.value == 8


def test_embedded_fsdp_block_legacy_integer_strategy(tmp_path):
    doc = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "num_processes": 8,
        "num_machines": 1,
        "fsdp_config": {"fsdp_sharding_strategy": 1},  # 1 -> FULL_SHARD
    }

    spec = adapt_accelerate(*_write(tmp_path, doc))

    assert spec.sharding.value == "FULL_SHARD"


# --- accelerate launch line overriding the referenced config file ---------


def test_accelerate_launch_num_processes_flag_overrides_the_file(tmp_path):
    (tmp_path / "default_config.yaml").write_text(
        yaml.safe_dump({"compute_environment": "LOCAL_MACHINE", "distributed_type": "MULTI_GPU", "num_processes": 8})
    )
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\naccelerate launch --config_file default_config.yaml --num_processes 16 train.py\n")

    spec = adapt_bare(str(script), str(tmp_path))

    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 16


def test_accelerate_launch_without_override_uses_the_file_value(tmp_path):
    (tmp_path / "default_config.yaml").write_text(
        yaml.safe_dump({"compute_environment": "LOCAL_MACHINE", "distributed_type": "MULTI_GPU", "num_processes": 8})
    )
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\naccelerate launch --config_file default_config.yaml train.py\n")

    spec = adapt_bare(str(script), str(tmp_path))

    assert spec.world_size.value == 8


def test_accelerate_launch_mixed_precision_flag_overrides_the_file(tmp_path):
    (tmp_path / "default_config.yaml").write_text(
        yaml.safe_dump(
            {
                "compute_environment": "LOCAL_MACHINE",
                "distributed_type": "MULTI_GPU",
                "num_processes": 8,
                "mixed_precision": "no",
            }
        )
    )
    script = tmp_path / "run.sh"
    script.write_text(
        "#!/bin/bash\naccelerate launch --config_file default_config.yaml --mixed_precision fp16 train.py\n"
    )

    spec = adapt_bare(str(script), str(tmp_path))

    assert spec.mixed_precision.value == "fp16"


def test_accelerate_launch_embedded_deepspeed_routes_through_shell_adapter(tmp_path):
    (tmp_path / "default_config.yaml").write_text(
        yaml.safe_dump(
            {
                "compute_environment": "LOCAL_MACHINE",
                "distributed_type": "DEEPSPEED",
                "num_processes": 8,
                "num_machines": 1,
                "deepspeed_config": {"zero_stage": 3, "offload_optimizer_device": "nvme"},
            }
        )
    )
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\naccelerate launch --config_file default_config.yaml train.py\n")

    spec = adapt_bare(str(script), str(tmp_path))

    assert spec.sharding.value == 3
    assert spec.zero_offload.value == {"optimizer": "nvme", "param": None}
    assert spec.world_size.value == 8
