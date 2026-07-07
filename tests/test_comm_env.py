"""Tests for the comm_env bucket: the full NCCL/CUDA comm-related env var
set, captured with provenance from every env source (shell export, image
baked env, k8s container.env), with cross-source conflicts noted and
LOC/PIX/PXB/PHB/SYS-or-numeric level vars normalized.
"""

from unittest.mock import patch

from traincheck.adapters.bare import adapt_bare
from traincheck.ir import COMM_ENV_VARS, Field, build_comm_env


def test_shell_export_overrides_image_baked_value_and_notes_conflict():
    image_env = {"NCCL_SOCKET_IFNAME": "ib0"}
    shell_env = {"NCCL_SOCKET_IFNAME": "eth0"}

    comm_env = build_comm_env([("image:base", image_env), ("shell", shell_env)])

    field = comm_env["NCCL_SOCKET_IFNAME"]
    assert field.status == "resolved"
    assert field.value == "eth0"
    assert field.source == "shell"
    assert "ib0" in field.reason
    assert "image:base" in field.reason


def test_single_source_var_has_no_conflict_reason():
    comm_env = build_comm_env([("shell", {"NCCL_SOCKET_IFNAME": "eth0"})])

    field = comm_env["NCCL_SOCKET_IFNAME"]
    assert field.status == "resolved"
    assert field.value == "eth0"
    assert field.reason == ""


def test_var_absent_from_every_source_reports_absent():
    comm_env = build_comm_env([("shell", {})])

    assert comm_env["NCCL_PROTO"].status == "absent"
    assert comm_env["NCCL_PROTO"].value is None


def test_all_listed_vars_captured_from_a_mixed_env():
    env = {
        "NCCL_ALGO": "Ring",
        "NCCL_IB_DISABLE": "0",
        "NCCL_NET_GDR_LEVEL": "PXB",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_P2P_LEVEL": "3",
        "NCCL_IB_HCA": "mlx5_0",
        "NCCL_PROTO": "Simple",
        "NCCL_DEBUG": "INFO",
        "NCCL_SHM_DISABLE": "0",
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_TIMEOUT": "1800",
    }

    comm_env = build_comm_env([("shell", env)])

    assert set(comm_env) == set(COMM_ENV_VARS)
    assert comm_env["NCCL_ALGO"].value == "Ring"
    assert comm_env["NCCL_IB_DISABLE"].value == 0
    # string form kept as-is (named distance, not int-cast)
    assert comm_env["NCCL_NET_GDR_LEVEL"].value == "PXB"
    assert comm_env["NCCL_SOCKET_IFNAME"].value == "eth0"
    assert comm_env["NCCL_P2P_DISABLE"].value == 1
    # numeric form normalized to int
    assert comm_env["NCCL_P2P_LEVEL"].value == 3
    assert comm_env["NCCL_IB_HCA"].value == "mlx5_0"
    assert comm_env["NCCL_PROTO"].value == "Simple"
    assert comm_env["NCCL_DEBUG"].value == "INFO"
    assert comm_env["NCCL_SHM_DISABLE"].value == 0
    assert comm_env["NCCL_IB_TIMEOUT"].value == 22
    assert comm_env["NCCL_TIMEOUT"].value == 1800
    assert all(f.status == "resolved" for f in comm_env.values())


def test_p2p_level_and_gdr_level_both_accept_named_or_numeric_form():
    named = build_comm_env([("shell", {"NCCL_P2P_LEVEL": "sys", "NCCL_NET_GDR_LEVEL": "loc"})])
    assert named["NCCL_P2P_LEVEL"].value == "SYS"
    assert named["NCCL_NET_GDR_LEVEL"].value == "LOC"

    numeric = build_comm_env([("shell", {"NCCL_P2P_LEVEL": "2", "NCCL_NET_GDR_LEVEL": "5"})])
    assert numeric["NCCL_P2P_LEVEL"].value == 2
    assert numeric["NCCL_NET_GDR_LEVEL"].value == 5


def test_no_sources_leaves_every_var_absent():
    comm_env = build_comm_env([])

    assert all(f.status == "absent" for f in comm_env.values())
    assert set(comm_env) == set(COMM_ENV_VARS)


def test_adapt_bare_reports_shell_export_winning_over_image_baked_env(tmp_path):
    script = tmp_path / "run.sh"
    script.write_text(
        "#!/bin/bash\n"
        "export NCCL_SOCKET_IFNAME=eth0\n"
        "srun --container-image=nvcr.io/nvidia/pytorch:24.01-py3 --gpu-bind=none \\\n"
        "    torchrun --nnodes=1 --nproc-per-node=8 train.py\n"
    )

    fake_image_fields = {
        "pin_status": "pinned_soft",
        "cuda": Field(value=None, status="absent"),
        "nccl": Field(value=None, status="absent"),
        "framework": Field(value=None, status="absent"),
        "env": {"NCCL_SOCKET_IFNAME": "ib0", "NCCL_IB_HCA": "mlx5_0"},
    }

    with patch("traincheck.adapters.bare.extract_image", return_value=fake_image_fields):
        spec = adapt_bare(str(script), base_dir=str(tmp_path))

    ifname = spec.comm_env["NCCL_SOCKET_IFNAME"]
    assert ifname.value == "eth0"
    assert ifname.source == "shell"
    assert "ib0" in ifname.reason

    # only set by the image, so it comes through unconflicted
    assert spec.comm_env["NCCL_IB_HCA"].value == "mlx5_0"
