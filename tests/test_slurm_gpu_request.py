"""Tests for the Slurm adapter's GPU-request flag matrix, space-separated
directive form, contiguity rule, and heterogeneous/array job handling.
"""

from traincheck.adapters.slurm import adapt_slurm


def _write(tmp_path, body: str, name: str = "train.sbatch"):
    script = tmp_path / name
    script.write_text(body)
    return str(script), str(tmp_path)


def test_gres_untyped_resolves_gpus_per_node(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --gres=gpu:4\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 4
    assert spec.gpu_type.status == "absent"
    assert spec.world_size.value == 8


def test_gres_typed_resolves_gpus_per_node_and_type(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --gres=gpu:a100:4\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpus_per_node.value == 4
    assert spec.gpu_type.value == "a100"


def test_gpus_total_untyped_divides_across_nodes(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=4\n#SBATCH --gpus=16\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpus_per_node.value == 4
    assert spec.world_size.value == 16


def test_gpus_total_typed_divides_and_carries_type(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --gpus=h100:8\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpus_per_node.value == 4
    assert spec.gpu_type.value == "h100"
    assert spec.world_size.value == 8


def test_gpus_per_node_untyped(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=3\n#SBATCH --gpus-per-node=5\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpus_per_node.value == 5
    assert spec.world_size.value == 15


def test_gpus_per_node_typed(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --gpus-per-node=v100:8\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpus_per_node.value == 8
    assert spec.gpu_type.value == "v100"


def test_gpus_per_task_times_ntasks_per_node(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --gpus-per-task=2\n#SBATCH --ntasks-per-node=4\necho hi\n",
    )
    spec = adapt_slurm(path, base)

    assert spec.gpus_per_node.value == 8


def test_gpus_per_socket_and_ntasks_per_gpu_parse_without_crashing(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --gpus-per-socket=2\n#SBATCH --ntasks-per-gpu=1\necho hi\n",
    )
    spec = adapt_slurm(path, base)

    # neither flag alone is enough to derive a per-node count
    assert spec.gpus_per_node.status == "absent"


def test_gpus_total_vs_gpus_per_node_give_different_world_sizes(tmp_path):
    total_path, total_base = _write(
        tmp_path, "#!/bin/bash\n#SBATCH --nodes=4\n#SBATCH --gpus=16\necho hi\n", name="total.sbatch"
    )
    per_node_path, per_node_base = _write(
        tmp_path, "#!/bin/bash\n#SBATCH --nodes=4\n#SBATCH --gpus-per-node=2\necho hi\n", name="per_node.sbatch"
    )

    total_spec = adapt_slurm(total_path, total_base)
    per_node_spec = adapt_slurm(per_node_path, per_node_base)

    assert total_spec.world_size.value == 16
    assert per_node_spec.world_size.value == 8
    assert total_spec.world_size.value != per_node_spec.world_size.value


def test_space_separated_directives_parse_the_same_as_equals_form(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes 3\n#SBATCH --gpus-per-node 5\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.nodes.value == 3
    assert spec.gpus_per_node.value == 5
    assert spec.world_size.value == 15


def test_gpus_per_node_and_gres_together_is_flagged_as_mutually_exclusive(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --gpus-per-node=4\n#SBATCH --gres=gpu:8\necho hi\n",
    )
    spec = adapt_slurm(path, base)

    # a deterministic value is still chosen (the direct per-node flag wins)...
    assert spec.gpus_per_node.value == 4
    # ...but the conflict is noted, not silently dropped
    assert "mutually exclusive" in spec.gpus_per_node.reason


def test_inconsistent_gpu_type_across_flags_is_noted(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --gres=gpu:a100:4\n#SBATCH --gpus=h100:4\necho hi\n",
    )
    spec = adapt_slurm(path, base)

    assert spec.gpu_type.value == "a100"
    assert "inconsistent" in spec.gpu_type.reason.lower()


def test_constraint_or_yields_a_set_of_possible_types(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --constraint=a100|h100\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpu_type.value == {"a100", "h100"}


def test_constraint_and_stays_a_single_compound_value(tmp_path):
    path, base = _write(tmp_path, "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --constraint=a100&ib\necho hi\n")
    spec = adapt_slurm(path, base)

    assert spec.gpu_type.value == "a100&ib"


def test_slurm_gpus_on_node_resolves_into_launcher_nproc(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "#SBATCH --nodes=2\n"
        "#SBATCH --gpus-per-node=4\n"
        "torchrun --nnodes=2 --nproc-per-node=$SLURM_GPUS_ON_NODE train.py\n",
    )
    spec = adapt_slurm(path, base)

    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 4


def test_sbatch_after_a_command_is_ignored(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n#SBATCH --nodes=2\necho starting\n#SBATCH --gpus-per-node=8\n",
    )
    spec = adapt_slurm(path, base)

    assert spec.nodes.status == "resolved"
    assert spec.nodes.value == 2
    assert spec.gpus_per_node.status == "absent"


def test_array_job_directive_does_not_crash(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n#SBATCH --array=0-9%2\n#SBATCH --nodes=1\n#SBATCH --gpus-per-node=2\necho hi\n",
    )
    spec = adapt_slurm(path, base)

    assert spec.nodes.value == 1
    assert spec.gpus_per_node.value == 2


def test_heterogeneous_job_does_not_crash_and_uses_the_primary_group(tmp_path):
    path, base = _write(
        tmp_path,
        "#!/bin/bash\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --gpus-per-node=2\n"
        "#SBATCH hetjob\n"
        "#SBATCH --nodes=3\n"
        "#SBATCH --gpus-per-node=8\n"
        "echo hi\n",
    )
    spec = adapt_slurm(path, base)

    assert spec.nodes.value == 1
    assert spec.gpus_per_node.value == 2
