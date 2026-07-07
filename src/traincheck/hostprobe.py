"""Probe the current host for facts that only exist at runtime.

driver/kernel version, OFED, and the GPU peermem kernel module can never
be read from a config file - they're properties of whatever machine
physically runs the job. `--probe-host` reads them from *this* machine
instead, which may or may not be the machine the job actually lands on
(you might be running traincheck from a login node or your laptop while
the job runs elsewhere) - so every resolved value is labeled with the
hostname it came from, and it's still up to the caller to judge whether
that host is representative.

`run_fn` is injectable (same pattern as extract_image's `inspect_fn`) so
tests never need a real nvidia-smi/ofed_info/lsmod on the test machine.
"""

import socket
import subprocess
from typing import Callable, Optional

from traincheck.ir import Field
from traincheck.validator import JobSpec

RunFn = Callable[[list], Optional[str]]

HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")


def probe_host_facts(spec: JobSpec, run_fn: Optional[RunFn] = None, hostname: Optional[str] = None) -> None:
    """Try to resolve each still-unknown HostEnv field by running the
    check command on this machine. Fields that are already resolved or
    absent are left untouched. Fields the probe can't determine (tool
    missing, command failed) stay unknown, with the reason updated to say
    why the probe itself came up empty.
    """
    run_fn = run_fn or _default_run
    source = f"host:{hostname or socket.gethostname()}"

    old_ids = {id(getattr(spec, name)) for name in HOST_ENV_FIELDS}

    _probe_driver_version(spec, run_fn, source)
    _probe_kernel_version(spec, run_fn, source)
    _probe_ofed_version(spec, run_fn, source)
    _probe_peermem_loaded(spec, run_fn, source)

    spec.meta.unresolved = [f for f in spec.meta.unresolved if id(f) not in old_ids]
    for name in HOST_ENV_FIELDS:
        field = getattr(spec, name)
        if field.status == "unknown":
            spec.meta.unresolved.append(field)


def _probe_driver_version(spec: JobSpec, run_fn: RunFn, source: str) -> None:
    if spec.driver_version.status != "unknown":
        return
    output = run_fn(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if output:
        spec.driver_version = Field(
            value=output.splitlines()[0].strip(), status="resolved", source=source, confidence=1.0
        )
    else:
        spec.driver_version = Field(
            value=None, status="unknown", reason="nvidia-smi unavailable or no GPU visible on this host"
        )


def _probe_kernel_version(spec: JobSpec, run_fn: RunFn, source: str) -> None:
    if spec.kernel_version.status != "unknown":
        return
    output = run_fn(["uname", "-r"])
    if output:
        spec.kernel_version = Field(value=output.strip(), status="resolved", source=source, confidence=1.0)
    else:
        spec.kernel_version = Field(value=None, status="unknown", reason="uname failed on this host")


def _probe_ofed_version(spec: JobSpec, run_fn: RunFn, source: str) -> None:
    if spec.ofed_version.status != "unknown":
        return
    output = run_fn(["ofed_info", "-s"])
    if output:
        spec.ofed_version = Field(value=output.strip(), status="resolved", source=source, confidence=1.0)
    else:
        spec.ofed_version = Field(
            value=None,
            status="unknown",
            reason="ofed_info unavailable; OFED may not be installed on this host",
        )


def _probe_peermem_loaded(spec: JobSpec, run_fn: RunFn, source: str) -> None:
    if spec.peermem_loaded.status != "unknown":
        return
    output = run_fn(["lsmod"])
    if output is None:
        spec.peermem_loaded = Field(
            value=None, status="unknown", reason="lsmod unavailable on this host (not Linux, or not on PATH)"
        )
        return
    loaded = "nvidia_peermem" in output or "nv_peer_mem" in output
    spec.peermem_loaded = Field(value=loaded, status="resolved", source=source, confidence=1.0)


def _default_run(cmd: list) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout
