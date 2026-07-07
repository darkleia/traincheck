"""Tests for the full torchrun launcher flag set: rdzv/master/node-rank
fields, elastic --nnodes ranges, host-dependent --nproc-per-node, the
implicit --max-restarts default, and the --standalone/--rdzv-endpoint
conflict note.
"""

from traincheck.extractors.shell import extract_shell
from traincheck.ir import build_launcher_fields

BASE_DIR = "."


def _launcher_fields(script_text: str):
    shell = extract_shell(script_text, base_dir=BASE_DIR)
    return build_launcher_fields(shell["launcher"], "shell")


def test_full_torchrun_line_resolves_every_rdzv_field():
    script = """\
#!/bin/bash
export NODE_RANK=2

torchrun \\
  --nnodes=4 \\
  --nproc-per-node=8 \\
  --node-rank=$NODE_RANK \\
  --master-addr=node0 \\
  --master-port=29500 \\
  --rdzv-backend=c10d \\
  --rdzv-id=job123 \\
  --rdzv-endpoint=node0:29500 \\
  --max-restarts=3 \\
  train.py
"""
    fields = _launcher_fields(script)

    assert fields["launcher_kind"].value == "torchrun"
    assert fields["launcher_nnodes"].status == "resolved"
    assert fields["launcher_nnodes"].value == 4
    assert fields["launcher_nnodes_min"].value == 4
    assert fields["launcher_nnodes_max"].value == 4
    assert fields["launcher_nproc_per_node"].status == "resolved"
    assert fields["launcher_nproc_per_node"].value == 8
    assert fields["launcher_node_rank"].status == "resolved"
    assert fields["launcher_node_rank"].value == 2
    assert fields["launcher_master_addr"].value == "node0"
    assert fields["launcher_master_port"].value == 29500
    assert fields["launcher_rdzv_backend"].value == "c10d"
    assert fields["launcher_rdzv_id"].value == "job123"
    assert fields["launcher_rdzv_endpoint"].value == "node0:29500"
    assert fields["launcher_max_restarts"].status == "resolved"
    assert fields["launcher_max_restarts"].value == 3
    assert fields["launcher_max_restarts"].reason == ""
    assert fields["launcher_standalone"].status == "absent"
    assert fields["world_size"].status == "resolved"
    assert fields["world_size"].value == 32


def test_elastic_nnodes_range_leaves_nnodes_and_world_size_unknown():
    script = "torchrun --nnodes=2:4 --nproc-per-node=8 train.py\n"

    fields = _launcher_fields(script)

    assert fields["launcher_nnodes_min"].value == 2
    assert fields["launcher_nnodes_max"].value == 4
    assert fields["launcher_nnodes"].status == "unknown"
    assert fields["launcher_nnodes"].reason == "elastic node range"
    assert fields["world_size"].status == "unknown"
    assert fields["world_size"].reason == "elastic node range"


def test_host_dependent_nproc_per_node_is_unknown_not_absent():
    script = "torchrun --nnodes=2 --nproc-per-node=gpu train.py\n"

    fields = _launcher_fields(script)

    assert fields["launcher_nnodes"].status == "resolved"
    assert fields["launcher_nnodes"].value == 2
    assert fields["launcher_nproc_per_node"].status == "unknown"
    assert fields["launcher_nproc_per_node"].reason == "per-node count is host-dependent"
    assert fields["world_size"].status == "unknown"
    assert fields["world_size"].reason == "per-node count is host-dependent"


def test_max_restarts_defaults_to_zero_when_torchrun_omits_it():
    script = "torchrun --nnodes=1 --nproc-per-node=8 train.py\n"

    fields = _launcher_fields(script)

    assert fields["launcher_max_restarts"].status == "resolved"
    assert fields["launcher_max_restarts"].value == 0
    assert fields["launcher_max_restarts"].reason == "torchrun default (no --max-restarts given)"


def test_standalone_flag_resolves_true_with_no_conflict_by_default():
    script = "torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py\n"

    fields = _launcher_fields(script)

    assert fields["launcher_standalone"].status == "resolved"
    assert fields["launcher_standalone"].value is True
    assert fields["launcher_standalone"].reason == ""


def test_standalone_with_explicit_rdzv_endpoint_notes_the_conflict():
    script = "torchrun --standalone --rdzv-endpoint=node0:29500 --nnodes=1 --nproc-per-node=8 train.py\n"

    fields = _launcher_fields(script)

    assert fields["launcher_standalone"].value is True
    assert fields["launcher_standalone"].reason == "standalone conflicts with explicit rendezvous endpoint"
