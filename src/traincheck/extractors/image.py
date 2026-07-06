"""Extract launcher-relevant signals from a container image reference.

Never pulls image layers - only the config blob (env vars), and only for
refs pinned enough to make that worth trusting. A floating reference
(`:latest` or no tag) can point at a different image tomorrow, so its
versions are reported unknown rather than inspected at all.
"""

import json
import subprocess
from typing import Any, Callable, Optional

from traincheck.ir import Field
from traincheck.utils import parse_version

InspectFn = Callable[[str], dict]

_CUDA_ENV_KEYS = ("CUDA_VERSION",)
_NCCL_ENV_KEYS = ("NCCL_VERSION",)
_FRAMEWORK_ENV_KEYS = ("NVIDIA_PYTORCH_VERSION", "PYTORCH_VERSION", "FRAMEWORK_VERSION")

_FLOATING_REASON = (
    "image reference is unpinned (no digest or specific tag); installed "
    "versions could change on the next pull, so nothing here can be trusted"
)


def extract_image(ref: str, inspect_fn: Optional[InspectFn] = None) -> dict:
    """Read cuda/nccl/framework versions out of an image reference.

    `inspect_fn(ref) -> dict` should return the parsed image config JSON
    (the same shape `skopeo inspect --config` produces, with `.config.Env`
    holding the "KEY=VALUE" environment list). Defaults to actually
    shelling out to skopeo; tests should inject a fake.
    """
    inspect_fn = inspect_fn or _default_inspect
    pin_status = _pin_status(ref)

    if pin_status == "floating":
        return _unknown_result(pin_status, _FLOATING_REASON)

    try:
        config = inspect_fn(ref)
    except Exception as exc:  # noqa: BLE001 - any inspect failure is reportable, not fatal
        return _unknown_result(pin_status, f"image inspect failed: {exc}")

    env = _parse_env(config)
    confidence = 1.0 if pin_status == "pinned_hard" else 0.7
    source = f"image:{ref}"

    return {
        "pin_status": pin_status,
        "cuda": _field_from_env(env, _CUDA_ENV_KEYS, source, confidence),
        "nccl": _field_from_env(env, _NCCL_ENV_KEYS, source, confidence),
        "framework": _field_from_env(env, _FRAMEWORK_ENV_KEYS, source, confidence),
        "env": env,
    }


def _unknown_result(pin_status: str, reason: str) -> dict:
    return {
        "pin_status": pin_status,
        "cuda": Field(value=None, status="unknown", reason=reason),
        "nccl": Field(value=None, status="unknown", reason=reason),
        "framework": Field(value=None, status="unknown", reason=reason),
        "env": {},
    }


def _pin_status(ref: str) -> str:
    ref_without_digest, _, digest = ref.partition("@")
    if digest:
        return "pinned_hard"

    tail = ref_without_digest.rsplit("/", 1)[-1]
    tag = tail.rsplit(":", 1)[1] if ":" in tail else None

    if tag is None or tag == "latest":
        return "floating"
    return "pinned_soft"


def _parse_env(config: dict) -> dict:
    entries = (config.get("config") or {}).get("Env") or []
    env = {}
    for entry in entries:
        if "=" in entry:
            key, value = entry.split("=", 1)
            env[key] = value
    return env


def _field_from_env(env: dict, keys: tuple, source: str, confidence: float) -> Field:
    for key in keys:
        if key in env:
            return Field(
                value=parse_version(env[key]), status="resolved", source=source, confidence=confidence
            )
    return Field(value=None, status="absent", source=source)


def _default_inspect(ref: str) -> Any:
    result = subprocess.run(
        ["skopeo", "inspect", "--config", f"docker://{ref}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)
