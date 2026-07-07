"""Tests for --probe-host's live host-fact probing."""

from pathlib import Path

from traincheck.adapters.slurm import adapt_slurm
from traincheck.hostprobe import HOST_ENV_FIELDS, probe_host_facts
from traincheck.ir import Field

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "slurm"


def _unknown_spec():
    return adapt_slurm(str(EXAMPLES_DIR / "train.sbatch"), base_dir=str(EXAMPLES_DIR))


def test_all_host_facts_start_unknown():
    spec = _unknown_spec()
    for name in HOST_ENV_FIELDS:
        assert getattr(spec, name).status == "unknown"
    assert len(spec.meta.unresolved) == 4


def test_successful_probe_resolves_all_four_fields():
    spec = _unknown_spec()

    fake_outputs = {
        "nvidia-smi": "535.129.03\n",
        "uname": "5.15.0-generic\n",
        "ofed_info": "MLNX_OFED_LINUX-5.8-1.0.1.1\n",
        "lsmod": "Module                  Size  Used by\nnvidia_peermem        16384  0\n",
    }

    def fake_run(cmd):
        return fake_outputs.get(cmd[0])

    probe_host_facts(spec, run_fn=fake_run, hostname="gpu-node-01")

    assert spec.driver_version.status == "resolved"
    assert spec.driver_version.value == "535.129.03"
    assert spec.driver_version.source == "host:gpu-node-01"

    assert spec.kernel_version.status == "resolved"
    assert spec.kernel_version.value == "5.15.0-generic"

    assert spec.ofed_version.status == "resolved"
    assert spec.ofed_version.value == "MLNX_OFED_LINUX-5.8-1.0.1.1"

    assert spec.peermem_loaded.status == "resolved"
    assert spec.peermem_loaded.value is True

    # all four now resolved, so none should remain in meta.unresolved
    assert spec.meta.unresolved == []


def test_peermem_not_loaded_is_resolved_false_not_unknown():
    spec = _unknown_spec()

    def fake_run(cmd):
        if cmd[0] == "lsmod":
            return "Module                  Size  Used by\nsome_other_module      16384  0\n"
        return None

    probe_host_facts(spec, run_fn=fake_run, hostname="gpu-node-01")

    assert spec.peermem_loaded.status == "resolved"
    assert spec.peermem_loaded.value is False


def test_failed_probe_leaves_field_unknown_with_updated_reason():
    spec = _unknown_spec()

    def fake_run(cmd):
        return None  # every tool "missing" on this host

    probe_host_facts(spec, run_fn=fake_run, hostname="gpu-node-01")

    for name in HOST_ENV_FIELDS:
        field = getattr(spec, name)
        assert field.status == "unknown"
        assert field.reason  # updated to explain *why* the probe failed

    assert len(spec.meta.unresolved) == 4


def test_partial_probe_only_removes_resolved_fields_from_unresolved():
    spec = _unknown_spec()

    def fake_run(cmd):
        return "535.129.03\n" if cmd[0] == "nvidia-smi" else None

    probe_host_facts(spec, run_fn=fake_run, hostname="gpu-node-01")

    assert spec.driver_version.status == "resolved"
    for name in ("kernel_version", "ofed_version", "peermem_loaded"):
        assert getattr(spec, name).status == "unknown"

    assert len(spec.meta.unresolved) == 3
    assert spec.driver_version not in spec.meta.unresolved


def test_probe_does_not_overwrite_an_already_resolved_field():
    spec = _unknown_spec()
    spec.driver_version = Field(value="999.99", status="resolved", source="test", confidence=1.0)

    def fake_run(cmd):
        return "535.129.03\n"

    probe_host_facts(spec, run_fn=fake_run, hostname="gpu-node-01")

    assert spec.driver_version.value == "999.99"
    assert spec.driver_version.source == "test"
