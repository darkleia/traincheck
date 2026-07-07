"""End-to-end coverage: resolve() dispatches to every adapter, across every
supported stack, and each result is genuinely usable (something resolves,
host facts needing a human check surface where they should).
"""

from pathlib import Path

import pytest

from traincheck.resolve import resolve
from traincheck.validator import Validator
from traincheck.verification import collect_needs_verification

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "examples"

_HOST_ENV_FIELD_NAMES = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")
_RESOURCE_OR_LAUNCHER_FIELDS = (
    "nodes",
    "gpus_per_node",
    "gpu_type",
    "world_size",
    "launcher_nnodes",
    "launcher_nproc_per_node",
)

# (label, entrypoint path, expected meta.stack value). torchx delegates to
# whichever scheduler it targets and reports *that* as meta.stack (here,
# "slurm"), not "torchx" itself - see adapt_torchx's own docstring/tests.
# The torchx entrypoint is component.py, not run.sh: detect_stack's own
# torchx signature is "python file that imports torchx", which run.sh
# (a shell script invoking the `torchx` CLI) doesn't match.
STACK_CASES = [
    ("slurm", EXAMPLES_ROOT / "slurm" / "train.sbatch", "slurm"),
    ("pbs", EXAMPLES_ROOT / "pbs" / "train.pbs", "pbs"),
    ("lsf", EXAMPLES_ROOT / "lsf" / "train.lsf", "lsf"),
    ("sge", EXAMPLES_ROOT / "sge" / "train.sge", "sge"),
    ("k8s_crd", EXAMPLES_ROOT / "k8s_crd" / "pytorchjob.yaml", "k8s_crd"),
    ("trainjob", EXAMPLES_ROOT / "trainjob" / "trainjob.yaml", "k8s_crd"),
    ("skypilot", EXAMPLES_ROOT / "skypilot" / "task.yaml", "skypilot"),
    ("accelerate", EXAMPLES_ROOT / "accelerate" / "default_config.yaml", "accelerate"),
    ("ray", EXAMPLES_ROOT / "ray" / "cluster.yaml", "ray"),
    ("bare", EXAMPLES_ROOT / "bare" / "run.sh", "bare"),
    ("torchx", EXAMPLES_ROOT / "torchx" / "component.py", "slurm"),
    ("submitit", EXAMPLES_ROOT / "submitit" / "job.py", "submitit"),
    ("native", EXAMPLES_ROOT / "native" / "job.traincheck.yaml", "native"),
]

# native's parser reports host facts as resolved-to-None (a native config
# document can, in principle, describe host expectations even if this one
# doesn't) rather than unknown - there's no live host in that world at
# all, so "go check this on the machine" doesn't apply the way it does for
# every launcher/script-based adapter. That's a deliberate, tested
# distinction from the original Field retrofit, not an oversight here.
STACKS_WITHOUT_HOST_FACT_FLAGS = {"native"}


@pytest.mark.parametrize("label,path,expected_stack", STACK_CASES, ids=[c[0] for c in STACK_CASES])
def test_resolve_reports_the_correct_meta_stack(label, path, expected_stack):
    spec = resolve(str(path))

    assert spec.meta.stack is not None
    assert spec.meta.stack.value == expected_stack


@pytest.mark.parametrize("label,path,expected_stack", STACK_CASES, ids=[c[0] for c in STACK_CASES])
def test_resolve_end_to_end_per_stack(label, path, expected_stack):
    spec = resolve(str(path))
    result = Validator().validate_spec(spec)

    resolved = [name for name in _RESOURCE_OR_LAUNCHER_FIELDS if getattr(spec, name).status == "resolved"]
    assert resolved, f"{label}: expected at least one Resources/Launcher field to resolve, got none"

    items = collect_needs_verification(spec, result)
    flagged_host_fields = {item.field_name for item in items} & set(_HOST_ENV_FIELD_NAMES)

    if label in STACKS_WITHOUT_HOST_FACT_FLAGS:
        assert not flagged_host_fields, f"{label}: host facts shouldn't be flagged for this stack"
    else:
        assert flagged_host_fields, f"{label}: expected host facts to surface in needs_verification"
