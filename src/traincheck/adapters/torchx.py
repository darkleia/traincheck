"""Map a torchx run invocation onto a JobSpec.

torchx is a launcher-of-launchers: `torchx run -s <scheduler> ...` picks a
*backend* scheduler (slurm, kubernetes, local_cwd, ...) and torchx itself
translates the component into whatever that backend expects. So once we
know which scheduler was targeted, the right move is to delegate to that
backend's own adapter for software/env facts - here, `adapt_slurm` - and
just merge its results in, while keeping torchx's own parsed job geometry
(`-j NxM`) rather than whatever (if anything) the delegate infers for it,
since torchx's own arguments are the more authoritative source for that.

If no scheduler can be determined at all - neither `-s <sched>` on the
run line nor a `[cli:run] scheduler` fallback in .torchxconfig - that's
reported via `meta.stack`, not guessed at.
"""

import configparser
import dataclasses
import re
from pathlib import Path
from typing import Optional

from traincheck.adapters.slurm import adapt_slurm
from traincheck.ir import Field, resolved_or_absent
from traincheck.validator import JobSpec

_BACKEND_ADAPTERS = {
    "slurm": adapt_slurm,
}

_RUN_LINE_RE = re.compile(r"\btorchx\s+run\b")
_SCHEDULER_FLAG_RE = re.compile(r"(?:^|\s)-s\s+(\S+)")
_GEOMETRY_FLAG_RE = re.compile(r"(?:^|\s)-j\s+(\S+)")
_GEOMETRY_VALUE_RE = re.compile(r"^(\d+)x(\d+)$")

_GEOMETRY_FIELDS = {
    "nodes",
    "gpus_per_node",
    "world_size",
    "launcher_nnodes",
    "launcher_nproc_per_node",
    "launcher_kind",
}


def adapt_torchx(path: str, base_dir: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    base = Path(base_dir)
    source = "torchx"
    run_line = _find_run_line(text)

    scheduler = _scheduler_from_run_line(run_line) if run_line else None
    if scheduler is None:
        scheduler = _scheduler_from_torchxconfig(base)

    nodes, nproc_per_node = _parse_geometry(run_line) if run_line else (None, None)
    if nodes is None or nproc_per_node is None:
        nodes, nproc_per_node = _geometry_from_torchxconfig(base)

    spec = JobSpec()
    spec.launcher_kind = resolved_or_absent("torchx", source)
    spec.launcher_nnodes = resolved_or_absent(nodes, source)
    spec.launcher_nproc_per_node = resolved_or_absent(nproc_per_node, source)
    world_size = nodes * nproc_per_node if nodes is not None and nproc_per_node is not None else None
    spec.world_size = resolved_or_absent(world_size, source)

    if scheduler is None:
        _mark_stack_unknown(
            spec,
            "no `-s <scheduler>` on the torchx run line and no [cli:run] scheduler in .torchxconfig",
        )
        return spec

    delegate = _BACKEND_ADAPTERS.get(scheduler)
    if delegate is None:
        _mark_stack_unknown(spec, f"torchx scheduler '{scheduler}' has no backend adapter yet")
        return spec

    delegate_spec = delegate(path, base_dir)
    _merge_delegate(spec, delegate_spec)
    spec.meta.stack = Field(value=scheduler, status="resolved", source=source, confidence=1.0)

    return spec


def _mark_stack_unknown(spec: JobSpec, reason: str) -> None:
    stack_field = Field(value=None, status="unknown", reason=reason)
    spec.meta.stack = stack_field
    spec.meta.unresolved.append(stack_field)


def _merge_delegate(spec: JobSpec, delegate_spec: JobSpec) -> None:
    for f in dataclasses.fields(spec):
        if f.name == "meta" or f.name in _GEOMETRY_FIELDS:
            continue
        setattr(spec, f.name, getattr(delegate_spec, f.name))
    spec.meta.unresolved.extend(delegate_spec.meta.unresolved)


def _find_run_line(text: str) -> Optional[str]:
    for line in text.splitlines():
        if _RUN_LINE_RE.search(line):
            return line
    return None


def _scheduler_from_run_line(line: str) -> Optional[str]:
    match = _SCHEDULER_FLAG_RE.search(line)
    return match.group(1) if match else None


def _parse_geometry(line: str) -> tuple:
    match = _GEOMETRY_FLAG_RE.search(line)
    if not match:
        return None, None
    return _parse_geometry_value(match.group(1))


def _parse_geometry_value(value: str) -> tuple:
    match = _GEOMETRY_VALUE_RE.match(value.strip())
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _scheduler_from_torchxconfig(base_dir: Path) -> Optional[str]:
    config = _read_torchxconfig(base_dir)
    if config is None:
        return None
    if config.has_section("cli:run") and config.has_option("cli:run", "scheduler"):
        return config.get("cli:run", "scheduler")
    return None


def _geometry_from_torchxconfig(base_dir: Path) -> tuple:
    config = _read_torchxconfig(base_dir)
    if config is None:
        return None, None
    for section in config.sections():
        if section.startswith("component:") and config.has_option(section, "j"):
            return _parse_geometry_value(config.get(section, "j"))
    return None, None


def _read_torchxconfig(base_dir: Path) -> Optional[configparser.ConfigParser]:
    path = base_dir / ".torchxconfig"
    if not path.is_file():
        return None
    config = configparser.ConfigParser()
    try:
        config.read(path)
    except configparser.Error:
        return None
    return config
