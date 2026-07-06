"""Tests for the container-image signal extractor."""

import json
from unittest.mock import patch

from traincheck.extractors.image import extract_image

FAKE_ENV = {
    "config": {
        "Env": [
            "PATH=/usr/bin",
            "CUDA_VERSION=12.2.128",
            "NCCL_VERSION=2.19.3",
            "NVIDIA_PYTORCH_VERSION=24.01",
        ]
    }
}


def fake_inspect(ref: str) -> dict:
    return FAKE_ENV


def test_digest_pinned_ref_reads_versions_at_high_confidence():
    ref = "nvcr.io/nvidia/pytorch@sha256:" + "a" * 64
    result = extract_image(ref, inspect_fn=fake_inspect)

    assert result["pin_status"] == "pinned_hard"
    assert result["cuda"].status == "resolved"
    assert result["cuda"].value == (12, 2, 128)
    assert result["cuda"].confidence == 1.0
    assert result["nccl"].value == (2, 19, 3)
    assert result["framework"].value == (24, 1)


def test_soft_tag_reads_versions_but_at_lower_confidence():
    ref = "nvcr.io/nvidia/pytorch:24.01-py3"
    result = extract_image(ref, inspect_fn=fake_inspect)

    assert result["pin_status"] == "pinned_soft"
    assert result["cuda"].status == "resolved"
    assert result["cuda"].value == (12, 2, 128)
    assert result["cuda"].confidence < 1.0


def test_latest_tag_is_floating_and_unknown_without_calling_inspect():
    def must_not_be_called(ref: str) -> dict:
        raise AssertionError("inspect_fn must not run for a floating ref")

    result = extract_image("nvcr.io/nvidia/pytorch:latest", inspect_fn=must_not_be_called)

    assert result["pin_status"] == "floating"
    for key in ("cuda", "nccl", "framework"):
        assert result[key].status == "unknown"
        assert result[key].reason


def test_bare_ref_with_no_tag_is_also_floating():
    def must_not_be_called(ref: str) -> dict:
        raise AssertionError("inspect_fn must not run for a floating ref")

    result = extract_image("nvcr.io/nvidia/pytorch", inspect_fn=must_not_be_called)

    assert result["pin_status"] == "floating"


def test_inspect_failure_yields_unknown_with_reason():
    def raises(ref: str) -> dict:
        raise RuntimeError("skopeo: manifest unknown")

    result = extract_image("nvcr.io/nvidia/pytorch:24.01-py3", inspect_fn=raises)

    assert result["pin_status"] == "pinned_soft"
    for key in ("cuda", "nccl", "framework"):
        assert result[key].status == "unknown"
        assert "skopeo: manifest unknown" in result[key].reason


def test_missing_env_var_is_absent_not_unknown():
    def sparse_inspect(ref: str) -> dict:
        return {"config": {"Env": ["PATH=/usr/bin"]}}

    result = extract_image("nvcr.io/nvidia/pytorch:24.01-py3", inspect_fn=sparse_inspect)

    assert result["cuda"].status == "absent"


def test_default_inspect_fn_wraps_skopeo_inspect_config():
    from traincheck.extractors.image import _default_inspect

    fake_completed = type("Completed", (), {"stdout": json.dumps(FAKE_ENV)})()
    with patch("subprocess.run", return_value=fake_completed) as mock_run:
        config = _default_inspect("nvcr.io/nvidia/pytorch:24.01-py3")

    mock_run.assert_called_once_with(
        ["skopeo", "inspect", "--config", "docker://nvcr.io/nvidia/pytorch:24.01-py3"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert config == FAKE_ENV
